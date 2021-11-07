# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.layers as L

from langml.plm.albert import load_albert
from langml.plm.bert import load_bert
from langml.layers import CRF
from langml.baselines import BaselineModel, Parameters
from langml.tensor_typing import Models


class BertCRF(BaselineModel):
    def __init__(self,
                 config_path: str,
                 ckpt_path: str,
                 params: Parameters,
                 backbone: str = 'roberta'):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.params = params
        assert backbone in ['bert', 'roberta', 'albert']
        if backbone == 'albert':
            self.load_plm = load_albert
        else:
            self.load_plm = load_bert

    def build_model(self, lazy_restore=False) -> Models:
        if lazy_restore:
            model, _, restore_bert_weights = self.load_plm(self.config_path, self.ckpt_path, lazy_restore=True)
        else:
            model, _ = self.load_plm(self.config_path, self.ckpt_path)
        crf = CRF(self.params.tag_size, sparse_target=False, name='crf')
        output = L.Dropout(self.params.dropout_rate)(model.output)
        output = L.Dense(self.params.tag_size, name='tag')(output)
        output = crf(output)
        train_model = keras.Model(model.input, output)
        train_model.summary()
        train_model.compile(keras.optimizers.Adam(self.params.learning_rate), loss=crf.loss, metrics=[crf.accuracy])

        if lazy_restore:
            restore_bert_weights(model)

        return train_model
