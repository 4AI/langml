# -*- coding: utf-8 -*-

from langml import keras
from langml.plm.layers import (
    TokenEmbedding, EmbeddingMatching, Masked,
)
from langml.plm.bert import load_bert  # NOQA
from langml.plm.albert import load_albert  # NOQA


custom_objects = {}
custom_objects.update(TokenEmbedding.get_custom_objects())
custom_objects.update(EmbeddingMatching.get_custom_objects())
custom_objects.update(Masked.get_custom_objects())

keras.utils.get_custom_objects().update(custom_objects)
