# -*- coding: utf-8 -*-

from langml import keras
from langml.prompt.models.ptuning import PartialEmbedding, PTuniningPrompt  # NOQA


custom_objects = {}
custom_objects.update(PartialEmbedding.get_custom_objects())

keras.utils.get_custom_objects().update(custom_objects)
