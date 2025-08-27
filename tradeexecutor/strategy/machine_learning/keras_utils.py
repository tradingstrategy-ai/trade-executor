"""Deal with Keras madness."""
import os
import random

import numpy as np

import tensorflow as tf


def make_keras_deterministic(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        # Cannot work after the first call
        # RuntimeError: Intra op parallelism cannot be modified after initialization.
        pass

