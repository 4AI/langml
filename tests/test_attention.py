# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras.backend as K
    import keras.layers as L


def test_self_attention_with_attn():
    from langml.layers import SelfAttention

    X = L.Input(shape=(None, 64))
    o, _ = SelfAttention(return_attention=True)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_self_attention_without_attn():
    from langml.layers import SelfAttention

    X = L.Input(shape=(None, 64))
    o = SelfAttention(return_attention=False)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_self_attention_with_mask():
    from langml.layers import SelfAttention

    X = L.Input(shape=(None, ))
    embed = L.Embedding(64, 64)(X)
    mask = L.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(X)
    o = SelfAttention(return_attention=False)(embed, mask=mask)
    assert K.int_shape(o) == K.int_shape(embed)


def test_self_additive_attention_with_attn():
    from langml.layers import SelfAdditiveAttention

    X = L.Input(shape=(None, 64))
    o, _ = SelfAdditiveAttention(return_attention=True)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_self_additive_attention_without_attn():
    from langml.layers import SelfAdditiveAttention

    X = L.Input(shape=(None, 64))
    o = SelfAdditiveAttention(return_attention=False)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_self_additive_attention_with_mask():
    from langml.layers import SelfAdditiveAttention

    X = L.Input(shape=(None, ))
    embed = L.Embedding(64, 64)(X)
    mask = L.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(X)
    o = SelfAdditiveAttention(return_attention=False)(embed, mask=mask)
    assert K.int_shape(o) == K.int_shape(embed)


def test_scaled_dot_product_attention():
    from langml.layers import ScaledDotProductAttention

    X = L.Input(shape=(None, 64))
    o, _ = ScaledDotProductAttention(return_attention=True)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_scaled_dot_product_attention_without_attn():
    from langml.layers import ScaledDotProductAttention

    X = L.Input(shape=(None, 64))
    o = ScaledDotProductAttention(return_attention=False)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_multihead_attention():
    from langml.layers import MultiHeadAttention

    X = L.Input(shape=(None, 64))
    o, _ = MultiHeadAttention(8, return_attention=True)(X)
    assert K.int_shape(o) == K.int_shape(X)


def test_multihead_attention_without_attn():
    from langml.layers import MultiHeadAttention

    X = L.Input(shape=(None, 64))
    o = MultiHeadAttention(8, return_attention=False)(X)
    assert K.int_shape(o) == K.int_shape(X)
