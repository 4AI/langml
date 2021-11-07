# -*- coding: utf-8 -*-

import os

import tensorflow as tf


__version__ = '0.0.1'

TF_VERSION = int(tf.__version__.split('.')[0])
TF_KERAS = int(os.getenv('TF_KERAS', 0)) == 1
