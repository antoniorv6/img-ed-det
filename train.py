seed_value = 8
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import models
import fire
import common

import efficientnet.tfkeras as efn
import numpy as np
from scipy.sparse import base
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa


physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.set_memory_growth(physical_devices[0], True)
except:
    pass
from tensorflow.keras import mixed_precision

from data import DataGenerator

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

import base_models
import models


@tf.function
def preprocess_image(img):
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1.0
    return img


def preprocess(mode, n_labels):
    @tf.function
    def fn(im1, im2, label):
        im1 = preprocess_image(im1)
        im2 = preprocess_image(im2)

        shape = tf.shape(label)
        batch = shape[0]

        if mode == 'sequence':
            ln_labels = n_labels + 1
            # tf.print(label)
            # label = tf.one_hot(label, depth=ln_labels, axis=-1)
            # print(labe)
            
            ed_seq = tf.concat([tf.ones((batch, 1), dtype=tf.int32)*ln_labels, label[:, :-1]], axis=1)
            y = {"label": label}

            x = {
                "original_image": im1,
                "edited_image": im2,
                "edition_sequence": ed_seq
            }
        else:
            x = {
                "original_image": im1,
                "edited_image": im2,
            }
        
            y = {"label": label[..., 0]}
        return x, y
    return fn


def get_dataset(generator, img_shape, label_shape, mode, n_labels):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(np.uint8, np.uint8, np.int32),
        output_shapes=((None,)+img_shape, (None,)+img_shape, (None,)+label_shape),
    )
    # dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess(mode, n_labels), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset


# def get_model(
#     n_classes,
#     input_shape=(224, 224, 3),
#     reuse_backbone=True,
#     output_activation="softmax",
# ):
#     orig_image = tf.keras.layers.Input(input_shape, name="original_image")
#     edit_image = tf.keras.layers.Input(input_shape, name="edited_image")
#     projection_dim = 512
#
#     backbone_fn = lambda name: tf.keras.models.Sequential(
#         [
#             efn.EfficientNetB0(include_top=False),
#             tf.keras.layers.Conv2D(projection_dim, 1, activation=tf.nn.gelu),
#             tf.keras.layers.Reshape((-1, projection_dim)),
#         ],
#         name=name,
#     )
#
#     backbone_fn = lambda name: tf.keras.models.Sequential(
#         [
#             ConvMixer(512, 12, input_shape=(224, 224, 6), kernel_size=8, patch_size=7),
#         ],
#         name=name,
#     )
#
#     backbone1 = backbone_fn(name="backbone1")
#     backbone2 = backbone1 if reuse_backbone else backbone_fn("backbone2")
#
#     # x = tf.keras.layers.Concatenate(axis=-1)([orig_image, edit_image])
#     # x = backbone1(x)
#     m1 = backbone1(orig_image)
#     m2 = backbone2(edit_image)
#
#     pos_encoding = tf.keras.layers.Embedding
#
#     num_patches = 7 * 7
#
#     positions = tf.range(start=0, limit=num_patches, delta=1)
#     pos_encoding = tf.keras.layers.Embedding(
#         input_dim=num_patches, output_dim=projection_dim
#     )(positions)
#     print(pos_encoding)
#
#     m1_enc = m1 + pos_encoding
#     m2_enc = m2 + pos_encoding
#
#     # pos_emb = RelativePositionalEmbedding(use_absolute_pos=False)
#     # m1_enc = pos_emb(m1)
#     # m2_enc = pos_emb(m2)
#     # print(m1_enc.shape)
#     # print(m1.shape)
#
#     # mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=projection_dim)
#     # m1 = mha(m1_enc, m2_enc)
#     # m2 = mha(m2_enc, m1)
#
#     # x = tf.keras.layers.Concatenate(axis=-1)([m1, m2])
#     # x = m1 - m2
#     # x = m1
#     # x = tf.keras.layers.Flatten()(x)
#     # x = tf.keras.layers.Concatenate(axis=-1)(
#     #     [
#     #         tf.keras.layers.GlobalAveragePooling2D()(x),
#     #         tf.keras.layers.GlobalMaxPooling2D()(x),
#     #     ]
#     # )
#
#     x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#
#     x = tf.keras.layers.Dense(n_classes, activation=output_activation, name="label")(x)
#     return tf.keras.models.Model(inputs=[orig_image, edit_image], outputs=[x])


def train(head='fcn', n_samples='single', seq_len='single', dataset='imagenette', repetitions=True, batch_size=16, load_pretrained=False):
    mode = 'next' if head == 'fcn' else 'sequence'
    # mode = 'next'

    datagen_params = DataGenerator.params_from_experiment(head=head, n_samples=n_samples, seq_len=seq_len, dataset=dataset, repetitions=repetitions)
    model_name = common.get_name(head=head, n_samples=n_samples, seq_len=seq_len, dataset=dataset, repetitions=repetitions)
    model_path = 'models/'+model_name
    log_path = 'logs/'+model_name+'.csv'

    if os.path.isdir(model_path) or os.path.isfile(model_path):
        if not load_pretrained:
            print(f"Model {model_name} already trained, skipping.")
            #return

    datagen_params['batch_size'] = batch_size
    train_gen = DataGenerator(split='train', shuffle=True, **datagen_params)
    val_gen = DataGenerator(split='val', shuffle=True, **datagen_params)

    train_dataset = get_dataset(train_gen.batch_generator, (None,)+train_gen.img_shape, (None,), mode=mode, n_labels=len(train_gen.distortions))
    val_dataset = get_dataset(val_gen.batch_generator, (None,)+val_gen.img_shape, (None,), mode=mode, n_labels=len(val_gen.distortions))
    val_dataset = val_dataset.take(val_gen.get_steps()).cache()

    lr = 1e-3
    if head == 'fcn':
        model_class = models.FC
        optimizer = tf.keras.optimizers.Adam
    elif head == 'rnn':
        model_class = models.RNN
        # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)
        optimizer = tf.keras.optimizers.Adam
    elif head == 'rnn_att':
        model_class = models.RNNAttention
        # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)
        optimizer = tf.keras.optimizers.Adam
    


    model = model_class(n_classes=len(train_gen.distortions), max_tokens=train_gen.max_n_distortions+1)
    # model.load(model_path)

    if (os.path.isdir(model_path) or os.path.isfile(model_path)) and load_pretrained:
        checkpoint = tf.train.Checkpoint(model.model)
        checkpoint.restore(model_path).expect_partial()
        # model.load(model_path)

    model.model.summary()
    model.model.compile(
        optimizer=optimizer(lr),
        loss=model.loss,
        metrics=model.metrics,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            constant_scheduler(decay_steps=[12, 20], decay_factor=0.1)
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1),
        tf.keras.callbacks.CSVLogger(
            log_path, separator=',', append=False
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    model.model.fit(
        train_dataset,
        epochs=150,
        validation_data=val_dataset,
        verbose=1,
        callbacks=callbacks,
        steps_per_epoch=train_gen.get_steps(),
        # validation_steps=val_gen.get_steps(),
    )


def constant_scheduler(decay_steps=[], decay_factor=0.1):
    def sch(epoch, lr):
        if epoch in decay_steps:
            lr *= decay_factor
        print(f"Learning rate for epoch {epoch}: {lr:.2}")
        return lr

    return sch

if __name__ == "__main__":
    fire.Fire(train)
