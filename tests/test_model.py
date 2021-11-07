# -*- coding: utf-8 -*-

import shutil

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.backend as K
    import keras.layers as L


def test_save_load_model_single_input():
    from langml.layers import SelfAttention
    from langml.model import save_frozen, load_frozen

    num_labels = 2
    embedding_size = 100
    hidden_size = 128

    model = keras.Sequential()
    model.add(L.Embedding(num_labels, embedding_size))
    model.add(L.Bidirectional(L.LSTM(hidden_size, return_sequences=True)))
    model.add(SelfAttention(hidden_size, return_attention=False))
    model.add(L.Dense(num_labels, activation='softmax'))
    model.compile('adam', loss='mse', metrics=['accuracy'])

    save_frozen(model, 'self_attn_frozen')
    K.clear_session()
    del model
    
    import tensorflow as tf
    tf_version = int(tf.__version__.split('.')[0])
    if tf_version > 1:
        model = load_frozen('self_attn_frozen')
    else:
        session = tf.Session(graph=tf.Graph())
        model = load_frozen('self_attn_frozen', session=session)
    shutil.rmtree('self_attn_frozen')
    assert model is not None


def test_save_load_model_multi_input():
    from langml.layers import SelfAttention
    from langml.model import save_frozen, load_frozen

    in1 = L.Input(shape=(None, 16), name='input-1')
    in2 = L.Input(shape=(None, 16), name='input-2')
    x1, x2 = in1, in2
    o1 = SelfAttention(return_attention=False)(x1)
    o2 = SelfAttention(return_attention=False)(x2)
    o = L.Concatenate()([o1, o2])
    o = L.Dense(2)(o)
    model = keras.Model([x1, x2], o)
    model.compile('adam', loss='mse', metrics=['accuracy'])

    save_frozen(model, 'self_attn_frozen.multi_input')
    K.clear_session()
    del model
    
    import tensorflow as tf
    tf_version = int(tf.__version__.split('.')[0])
    if tf_version > 1:
        model = load_frozen('self_attn_frozen.multi_input')
    else:
        session = tf.Session(graph=tf.Graph())
        model = load_frozen('self_attn_frozen.multi_input', session=session)
    shutil.rmtree('self_attn_frozen.multi_input')
    assert model is not None


def test_save_load_model_multi_input_output():
    from langml.layers import SelfAttention
    from langml.model import save_frozen, load_frozen

    in1 = L.Input(shape=(None, 16), name='input-1')
    in2 = L.Input(shape=(None, 16), name='input-2')
    x1, x2 = in1, in2
    o1 = SelfAttention(return_attention=False)(x1)
    o2 = SelfAttention(return_attention=False)(x2)
    model = keras.Model([x1, x2], [o1, o2])
    model.compile('adam', loss='mse', metrics=['accuracy'])

    save_frozen(model, 'self_attn_frozen.multi_input_output')
    K.clear_session()
    del model
    
    import tensorflow as tf
    tf_version = int(tf.__version__.split('.')[0])
    if tf_version > 1:
        model = load_frozen('self_attn_frozen.multi_input_output')
    else:
        session = tf.Session(graph=tf.Graph())
        model = load_frozen('self_attn_frozen.multi_input_output', session=session)
    shutil.rmtree('self_attn_frozen.multi_input_output')
    assert model is not None


def test_crf_save_load():
    from langml.layers import CRF
    from langml.model import save_frozen, load_frozen

    num_labels = 10
    embedding_size = 100
    hidden_size = 128

    model = keras.Sequential()
    model.add(L.Embedding(num_labels, embedding_size, mask_zero=True))
    model.add(L.LSTM(hidden_size, return_sequences=True))
    model.add(L.Dense(num_labels))
    crf = CRF(num_labels, sparse_target=False)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss, metrics=[crf.accuracy])

    save_frozen(model, 'crf_frozen')
    K.clear_session()
    del model
    
    import tensorflow as tf
    tf_version = int(tf.__version__.split('.')[0])
    if tf_version > 1:
        model = load_frozen('crf_frozen')
    else:
        session = tf.Session(graph=tf.Graph())
        model = load_frozen('crf_frozen', session=session)
    shutil.rmtree('crf_frozen')
    assert model is not None


def test_crf_dense_target_save_load():
    from langml.layers import CRF
    from langml.model import save_frozen, load_frozen

    num_labels = 10
    embedding_size = 100
    hidden_size = 128

    model = keras.Sequential()
    model.add(L.Embedding(num_labels, embedding_size, mask_zero=True))
    model.add(L.LSTM(hidden_size, return_sequences=True))
    model.add(L.Dense(num_labels))
    crf = CRF(num_labels, sparse_target=False)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss, metrics=[crf.accuracy])

    save_frozen(model, 'crf_frozen_dense_target')
    K.clear_session()
    del model

    import tensorflow as tf
    tf_version = int(tf.__version__.split('.')[0])

    if tf_version > 1:
        model = load_frozen('crf_frozen_dense_target')
    else:
        session = tf.Session(graph=tf.Graph())
        model = load_frozen('crf_frozen_dense_target', session=session)
    shutil.rmtree('crf_frozen_dense_target')
    assert model is not None
