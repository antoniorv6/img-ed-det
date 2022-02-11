import numpy as np
import efficientnet.tfkeras as efn
import tensorflow as tf
import tensorflow.keras.backend as K
from classification_models.tfkeras import Classifiers
ResNet50, preprocess_input = Classifiers.get('resnet50')

def preprocess(img):
    return img[np.newaxis, ...].astype(np.float32) / 127.5 - 1.0

def crossentropy_masked(y_true, y_pred):
    end_token = y_true[0, -1]
    # y_true = y_true[:, :1]
    # y_pred = y_pred[:, :1, :]

    mask = y_true != end_token
    mask = tf.cast(mask, y_pred.dtype)
    # y_true = K.argmax(y_true, axis=-1)
    batch = tf.shape(y_true)[0]
    mask = tf.concat([tf.ones((batch, 1), dtype=y_pred.dtype), mask[:, :-1]], axis=1)

    loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
    loss = K.sum(loss * mask, axis=-1) / K.sum(mask, axis=-1)

    return loss

def acc_m(y_true, y_pred):
    mask = y_true != y_true[0, -1]
    mask = tf.cast(mask, tf.int32)
    batch = tf.shape(y_true)[0]
    mask = tf.concat([tf.ones((batch, 1), dtype=tf.int32), mask[:, :-1]], axis=1)

    y_true = tf.cast(y_true, tf.int32) * mask
    y_pred = tf.cast(K.argmax(y_pred, axis=-1), tf.int32) * mask

    acc = tf.math.reduce_all(y_true == y_pred, axis=-1)
    return K.mean(acc)

def s1_acc(y_true, y_pred):
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1, :]
    # y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    acc = tf.cast(tf.cast(y_true, tf.int32) == tf.cast(y_pred, tf.int32), tf.float32)
    return K.mean(acc)

def s0_acc(y_true, y_pred):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0, :]
    # y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    acc = tf.cast(tf.cast(y_true, tf.int32) == tf.cast(y_pred, tf.int32), tf.float32)
    return K.mean(acc)

def ms_acc(y_true, y_pred):
    y_true == y_pred
    mask = y_true[:, :, -1] != 1
    mask = tf.cast(mask, tf.int64)
    batch = tf.shape(y_true)[0]
    mask = tf.concat([tf.ones((batch, 1), dtype=tf.int64), mask[:, 1:]], axis=1)

    labels = K.argmax(y_true, axis=-1)
    preds = K.argmax(y_pred, axis=-1)
    acc = tf.cast(tf.math.reduce_all((preds * mask) == (labels * mask), axis=-1), tf.float32)

    return K.mean(acc)


def get_basic_fusion_model(
    backbone,
    orig_image_input,
    edit_image_input,
    fuse_mode="sub",
):
    fusions = {
        "sub": tf.subtract,
        "add": tf.add,
        "mult": tf.multiply,
        "concat": lambda a, b: tf.concat([a, b], axis=-1),
    }
    assert (
        fuse_mode in fusions.keys()
    ), f"Unknown fuse mode. Posible fuse modes: {list(fusions.keys())}"

    x_o = backbone(orig_image_input)
    x_e = backbone(edit_image_input)

    x = fusions[fuse_mode](x_o, x_e)

    x = tf.keras.layers.Concatenate(axis=-1)(
        [
            tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x),
            tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalMaxPooling2D())(x),
        ]
    )
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.GlobalAveragePooling1D('channels_last')(x)

    return tf.keras.models.Model(
        inputs=[orig_image_input, edit_image_input], outputs=[x]
    )


class Model():
    def __init__(self):
        self.original_image_input = tf.keras.layers.Input(shape=(None, 224, 224, 3), name="original_image")
        self.transformed_image_input = tf.keras.layers.Input(shape=(None, 224, 224, 3), name="edited_image")

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    def predict(self, original_image, transformed_image, max_steps=1):
        raise NotImplementedError()

class FC(Model):
    def __init__(self, *, n_classes, output_activation='softmax', max_tokens, **kwargs):
        super().__init__(**kwargs)

        # self.backbone = efn.EfficientNetB0(include_top=False)
        self.backbone = ResNet50((224, 224, 3), weights='imagenet', include_top=False)
        self.backbone = tf.keras.layers.TimeDistributed(self.backbone)

        self.base_model = get_basic_fusion_model(
            backbone=self.backbone,
            orig_image_input=self.original_image_input,
            edit_image_input=self.transformed_image_input,
            fuse_mode="sub",
        )

        self.n_classes = n_classes
        self.output_activation = output_activation
        self.max_tokens = max_tokens

        self.model = self.build()
    
    def predict(self, *, original_image, transformed_image, transformation_fns=[], max_steps=1):
        curr_image = original_image
        predicted_sequence = [self.n_classes for _ in range(max_steps)]
        predicted_probs = np.zeros((max_steps, self.n_classes+1))
        predicted_probs[:, self.n_classes] = 1
        for step in range(max_steps):
            pred = self.model({'original_image': preprocess(curr_image), 'edited_image': preprocess(transformed_image)})
            predicted_probs[step] = np.squeeze(pred)
            pred = np.squeeze(np.argmax(pred, axis=-1))
            predicted_sequence[step] = pred

            for i in range(curr_image.shape[0]):
                curr_image[i] = transformation_fns[pred](curr_image[i])

            if np.all(curr_image == transformed_image):
                break
                # predicted_sequence.append(self.n_classes)
                # predicted_sequence += ([self.n_classes] * (max_steps - len(predicted_sequence)))
        return np.array(predicted_sequence), predicted_probs

    
    def build(self):
        head = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(128, activation=tf.nn.gelu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation=tf.nn.gelu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(
                    self.n_classes, activation=self.output_activation, name="label"
                ),
            ],
            name="label",
        )
        out = head(self.base_model.outputs[0])
        return tf.keras.models.Model([self.original_image_input, self.transformed_image_input], out)

    def load(self, path):
        self.model.load_weights(path)

class Decoder(tf.keras.models.Model):
    def __init__(self, *, embedding_dim, units, n_classes, use_attention, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.embedding = tf.keras.layers.Embedding(n_classes+1, embedding_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        if use_attention:
            self.attention = tf.keras.layers.Attention()
        else:
            self.attention = None


        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc0 = tf.keras.layers.Dense(units, activation=tf.nn.gelu)
        self.dr = tf.keras.layers.Dropout(0.3)
        self.fc1 = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        x, features, state = inputs
        x = self.embedding(x)

        # x = tf.concat([tf.expand_dims(features, 1), x], axis=-1)
        # x = features

        x, state = self.gru(x, initial_state=tf.cast(state, tf.float16))

        if self.attention is not None:
            state = self.attention([state, features, features])

        x = tf.squeeze(x, 1)
        x = self.fc0(x)
        x = self.layer_norm(x)
        x = self.dr(x)
        x = self.fc1(x)
        return x, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
class RNN(Model):
    def __init__(self, *, n_classes, max_tokens, output_activation='softmax', gru_units=264, use_attention=False, **kwargs):
        super().__init__(**kwargs)

        # self.backbone = efn.EfficientNetB0(include_top=False)
        self.backbone = ResNet50((224, 224, 3), weights='imagenet', include_top=False)
        self.backbone = tf.keras.layers.TimeDistributed(self.backbone)

        self.base_model = get_basic_fusion_model(
            backbone=self.backbone,
            orig_image_input=self.original_image_input,
            edit_image_input=self.transformed_image_input,
            fuse_mode="sub",
        )

        self.n_classes = n_classes
        self.output_activation = output_activation
        self.max_tokens = max_tokens
        self.gru_units = gru_units
        self.use_attention = use_attention

        self.embedding_size = self.base_model.outputs[0].shape[-1]

        self.model = self.build()

        self.metrics = [s0_acc, s1_acc, acc_m]
        self.loss = crossentropy_masked


    def predict(self, *, original_image, transformed_image, transformation_fns=[], max_steps=1):
        probs = self.model({'original_image': preprocess(original_image), 'edited_image': preprocess(transformed_image)})
        probs = np.squeeze(probs)
        pred = np.argmax(probs, axis=-1)
        return pred, probs
    
    def load(self, path):
        model = tf.keras.models.load_model(path, custom_objects={"acc_m": acc_m, "s0_acc": s0_acc, "s1_acc": s1_acc, "crossentropy_masked": crossentropy_masked})
        self.model.set_weights(model.get_weights())

    def build(self):
        inp = tf.keras.layers.Input(shape=(self.embedding_size,), name='embedding')

        emb = tf.keras.layers.Dense(self.gru_units)(inp)
        # emb = inp
        # feat_map = tf.keras.layers.Reshape((7, 7, self.embedding_size))

        decoder = Decoder(embedding_dim=128, units=self.gru_units, n_classes=self.n_classes+1, use_attention=self.use_attention)

        x_pred = []
        batch_size = tf.shape(emb)[0]
        state = emb
        x = tf.expand_dims(emb, 1)

        for i in range(self.max_tokens):
            x, state = decoder([tf.expand_dims(tf.zeros(tf.shape(state)[0]), 1) + (1 if i != 0 else 0), emb, state])
            # x, state = decoder([tf.expand_dims(inp_seq[..., i], 1), emb, state])

            out = tf.keras.layers.Activation('softmax')(x)
            x_pred.append(out)
        x_pred = tf.stack(x_pred, 1)

        head = tf.keras.models.Model([inp], x_pred, name='label')
        head.summary()

        train_model = tf.keras.models.Model([
            self.original_image_input,
            self.transformed_image_input,
            # inp_seq
        #], outputs=[head([self.base_model.outputs[0], inp_seq])])
        ], outputs=[head(self.base_model.outputs[0])])

        # pred_model = tf.keras.models.Model([
        #     self.original_image_input,
        #     self.transformed_image_input,
        # ], outputs=[pred_head(self.backbone.outputs[0], inp_seq)])

        return train_model


class RNNAttention(RNN):
    def __init__(self, **kwargs):
        super().__init__(use_attention=True, **kwargs)