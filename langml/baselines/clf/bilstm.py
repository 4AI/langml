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

from langml.baselines import BaselineModel, Parameters
from langml.layers import SelfAttention
from langml.tensor_typing import Models


class BiLSTMClassifier(BaselineModel):
    def __init__(self, params: Parameters, with_attention: bool = False):
        self.params = params
        self.with_attention = with_attention

    def build_model(self) -> Models:
        x_in = L.Input(shape=(None, ), name='Input-Token')
        x = x_in
        x = L.Embedding(self.params.vocab_size,
                        self.params.embedding_size,
                        mask_zero=True,
                        name='embedding')(x)
        if self.with_attention:
            x = L.Bidirectional(L.LSTM(self.params.hidden_size, return_sequences=True))(x)
            # attn
            x = SelfAttention()(x)
            x = L.Lambda(lambda x: K.max(x, 1))(x)
        else:
            x = L.Bidirectional(L.LSTM(self.params.hidden_size))(x)
        x = L.Dropout(0.2)(x)
        o = L.Dense(self.params.tag_size, name='tag', activation='softmax')(x)
        model = keras.Model(x_in, o)
        model.summary()
        model.compile(keras.optimizers.Adam(self.params.learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
