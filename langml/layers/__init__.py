# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
else:
    import keras

from langml.layers.crf import CRF
from langml.layers.layer_norm import LayerNorm
from langml.layers.layers import (
    AbsolutePositionEmbedding, SineCosinePositionEmbedding, ScaleOffset, ConditionalLayerNormalization
)
from langml.layers.attention import (
    SelfAttention, SelfAdditiveAttention,
    ScaledDotProductAttention, MultiHeadAttention,
    GatedAttentionUnit
)

custom_objects = {}
custom_objects.update(AbsolutePositionEmbedding.get_custom_objects())
custom_objects.update(SineCosinePositionEmbedding.get_custom_objects())
custom_objects.update(ScaleOffset.get_custom_objects())
custom_objects.update(ConditionalLayerNormalization.get_custom_objects())
custom_objects.update(CRF.get_custom_objects())
custom_objects.update(SelfAttention.get_custom_objects())
custom_objects.update(SelfAdditiveAttention.get_custom_objects())
custom_objects.update(LayerNorm.get_custom_objects())
custom_objects.update(ScaledDotProductAttention.get_custom_objects())
custom_objects.update(MultiHeadAttention.get_custom_objects())
custom_objects.update(GatedAttentionUnit.get_custom_objects())

keras.utils.get_custom_objects().update(custom_objects)
