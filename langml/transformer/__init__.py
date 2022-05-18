# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
else:
    import keras

from langml.transformer.layers import FeedForward


custom_objects = {}
custom_objects.update(FeedForward.get_custom_objects())

keras.utils.get_custom_objects().update(custom_objects)
