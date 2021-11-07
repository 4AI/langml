# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
else:
    import keras

from langml.layers.crf import CRF
from langml.layers.attention import (
    SelfAttention, SelfAdditiveAttention,
    ScaledDotProductAttention, MultiHeadAttention
)
from langml.layers.layer_norm import LayerNorm

custom_objects = {}
custom_objects.update(CRF.get_custom_objects())
custom_objects.update(SelfAttention.get_custom_objects())
custom_objects.update(SelfAdditiveAttention.get_custom_objects())
custom_objects.update(LayerNorm.get_custom_objects())
custom_objects.update(ScaledDotProductAttention.get_custom_objects())
custom_objects.update(MultiHeadAttention.get_custom_objects())

keras.utils.get_custom_objects().update(custom_objects)
