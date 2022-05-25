# -*- coding: utf-8 -*-

import os

import tensorflow as tf


__version__ = '0.4.1'

TF_VERSION = int(tf.__version__.split('.')[0])
TF_KERAS = int(os.getenv('TF_KERAS', 0)) == 1

if TF_KERAS:
    import tensorflow.keras as keras  # NOQA
    import tensorflow.keras.layers as L  # NOQA
    import tensorflow.keras.backend as K  # NOQA
else:
    import keras  # NOQA
    import keras.layers as L  # NOQA
    import keras.backend as K  # NOQA
