# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras.backend as K
    import keras.layers as L


def test_transformer_encoder():
    from langml.transformer.encoder import TransformerEncoder

    X = L.Input(shape=(None, 64))
    o = TransformerEncoder(4, 64)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_transformer_encoder_block():
    from langml.transformer.encoder import TransformerEncoderBlock

    X = L.Input(shape=(None, 64))
    o = TransformerEncoderBlock(2, 4, 64)(X)
    assert K.int_shape(o) == K.int_shape(X)
