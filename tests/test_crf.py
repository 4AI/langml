# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.backend as K
    import keras.layers as L


def test_crf():
    from langml.layers import CRF

    num_labels = 10
    embedding_size = 100
    hidden_size = 128

    X = L.Input(shape=(None, ), name='Input-X')
    embed = L.Embedding(num_labels, embedding_size, mask_zero=True)(X)
    encoded = L.Bidirectional(L.LSTM(hidden_size, return_sequences=True))(embed)
    output = L.Dense(num_labels)(encoded)
    crf = CRF(num_labels, sparse_target=True)
    output = crf(output)
    assert len(K.int_shape(output)) == 3


def test_crf_dense_target():
    from langml.layers import CRF

    num_labels = 10
    embedding_size = 100
    hidden_size = 128

    X = L.Input(shape=(None, ), name='Input-X')
    embed = L.Embedding(num_labels, embedding_size, mask_zero=True)(X)
    encoded = L.Bidirectional(L.LSTM(hidden_size, return_sequences=True))(embed)
    output = L.Dense(num_labels)(encoded)
    crf = CRF(num_labels, sparse_target=False)
    output = crf(output)
    assert len(K.int_shape(output)) == 3
