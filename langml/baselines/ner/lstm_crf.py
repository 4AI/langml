# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.layers as L

from langml.layers import CRF
from langml.baselines import BaselineModel, Parameters
from langml.tensor_typing import Models


class LSTMCRF(BaselineModel):
    def __init__(self, params: Parameters):
        self.params = params

    def build_model(self) -> Models:
        crf = CRF(self.params.tag_size, sparse_target=False)
        model = keras.Sequential()
        model.add(L.Embedding(self.params.vocab_size, self.params.embedding_size, mask_zero=False))
        model.add(L.Bidirectional(L.LSTM(self.params.hidden_size, return_sequences=True)))
        model.add(L.Dropout(self.params.dropout_rate))
        model.add(L.Dense(self.params.tag_size, name='tag'))
        model.add(crf)
        model.summary()
        model.compile(keras.optimizers.Adam(self.params.learning_rate), loss=crf.loss, metrics=[crf.accuracy])

        return model
