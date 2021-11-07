# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras.backend as K
    import keras.layers as L


def test_bert_encoder():
    from langml.plm.bert import BERT
    bert = BERT(
        100,
        embedding_dim=128, 
        transformer_blocks=2,
        attention_heads=2,
        intermediate_size=1000
    )
    bert.build()
    model = bert()
    assert K.int_shape(model.output) == (None, 512, 128)
