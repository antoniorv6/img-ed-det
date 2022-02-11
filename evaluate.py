seed_value = 1
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_seed(seed_value)
import progressbar

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
# from models.layers import RelativePositionalEmbedding
import distortions

import pyter
# from models.convmixer import ConvMixer


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
from train import get_dataset


def eval(head='fcn', n_samples='single', seq_len='single', dataset='imagenette', repetitions=True, test_size=None, out_name=None):

    ### Types ###
    # By head:
    # - FCN (next)
    # - Seq (multiple_sorted)
    #
    # By seq len:
    # - 1 (single)
    # - N (multiple_sorted)
    #
    # By dataset:
    # - Imagenette
    # - GDL
    #
    # By N samples:
    # - 1
    # - N
    #
    # By repetitions:
    # - Allow (max iters)
    # - Disallow
    mode = 'next' if head == 'fcn' else 'sequence'

    datagen_params = DataGenerator.params_from_experiment(head=head, n_samples=n_samples, seq_len=seq_len, dataset=dataset, repetitions=repetitions)
    model_name = out_name or common.get_name(head=head, n_samples=n_samples, seq_len=seq_len, dataset=dataset, repetitions=repetitions)
    model_path = 'models/'+model_name
    out_path = 'results/'+model_name+'.pk'
    log_path = 'logs/'+model_name+'.csv'

    if os.path.isfile(out_path):
        #return
        pass
    else:
        f = open(out_path, 'a')
        f.close()

    datagen_params['batch_size'] = 1
    # datagen_params['max_n_distortions'] = 3
    test_gen = DataGenerator(split='test', shuffle=True, **datagen_params)

    test_dataset = get_dataset(test_gen.batch_generator, (None,)+test_gen.img_shape, (None,), mode=mode, n_labels=len(test_gen.distortions))
    test_dataset = test_dataset.take(test_gen.get_steps()).cache()

    if head == 'fcn':
        model_class = models.FC
    elif head == 'rnn':
        model_class = models.RNN
    elif head == 'rnn_att':
        model_class = models.RNNAttention
    


    model = model_class(n_classes=len(test_gen.distortions), max_tokens=test_gen.max_n_distortions+1)
    # model.model = tf.keras.models.load_model(model_path)
    model.load(model_path)
    model.model.summary()
    distortion_fns = [getattr(distortions, d) for d in test_gen.distortions]
    print(distortion_fns)

    y_trues = []
    y_preds = []
    y_probs = []
    ious = []
    ters = []

    for i in progressbar.progressbar(range(test_size or test_gen.get_size())):
        #if i > 10:
        #    break
        orig_img, tform_img, label = test_gen.generate_image()
        pred, probs = model.predict(original_image=orig_img, transformed_image=tform_img, transformation_fns=distortion_fns, max_steps=test_gen.max_n_distortions+1)
        y_trues.append(label)
        y_preds.append(pred)
        y_probs.append(probs)
        ious.append(common.iou(label, pred))
        ters.append(common.ter(label, pred))

    print('res', np.stack(ters, axis=0).mean())
    # data = {"y_true": np.stack(y_trues, axis=0), "y_pred": np.stack(y_preds, axis=0), "iou": np.stack(ious, axis=0), "ter": np.stack(ters, axis=0)}
    data = {
        "y_true": np.stack(y_trues, axis=0),
        "y_pred": np.stack(y_preds, axis=0),
        "y_prob": np.stack(y_probs, axis=0),
        "iou": np.stack(ious, axis=0),
        "ter": np.stack(ters, axis=0)
    }

    import pickle
    with open(out_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    fire.Fire(eval)
