# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.layers as L

from langml.baselines import BaselineModel, Parameters
from langml.tensor_typing import Models


class TextCNNClassifier(BaselineModel):
    def __init__(self, params: Parameters):
        self.params = params

    def build_model(self) -> Models:
        x_in = L.Input(shape=(None, ), name='Input-Token')
        x = x_in
        x = L.Embedding(self.params.vocab_size,
                        self.params.embedding_size,
                        name='embedding')(x)
        convs = []
        for kernel_size in [3, 4, 5]:
            conv = L.Conv1D(filters=self.params.filter_size,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            activation='relu')(x)
            conv = L.MaxPooling1D()(conv)
            conv = L.GlobalMaxPool1D()(conv)
            convs.append(conv)
        x = L.Concatenate()(convs)
        x = L.Dropout(0.2)(x)
        o = L.Dense(self.params.tag_size, name='tag', activation='softmax')(x)
        model = keras.Model(x_in, o)
        model.summary()
        model.compile(keras.optimizers.Adam(self.params.learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
