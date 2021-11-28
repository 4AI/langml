# -*- coding: utf-8 -*-


""" Implementation P-Tuning

Paper: GPT Understands, Too
URL: https://arxiv.org/pdf/2103.10385.pdf
"""

from typing import List, Optional, Union

import numpy as np

from langml import L, K, keras
from langml.prompt.base import BasePromptModel, Template
from langml.tensor_typing import Constraint, Initializer, Models, Regularizer, Tensors


class PartialEmbedding(L.Embedding):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 active_start: int,
                 active_end: int,
                 embeddings_initializer: Optional[Initializer] = 'uniform',
                 embeddings_regularizer: Optional[Regularizer] = None,
                 activity_regularizer: Optional[Regularizer] = None,
                 embeddings_constraint: Optional[Constraint] = None,
                 mask_zero: bool = False,
                 input_length: Optional[int] = None,
                 **kwargs):
        self.active_start = active_start
        self.active_end = active_end
        super().__init__(
            input_dim, output_dim, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length, **kwargs)

    @staticmethod
    def get_custom_objects() -> dict:
        return {'PartialEmbedding': PartialEmbedding}

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> List[Union[Tensors, None]]:
        return [super(PartialEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs: Tensors) -> List[Tensors]:
        # https://stackoverflow.com/a/43368518
        mask = np.zeros((K.int_shape(self.embeddings)[0], 1))
        mask[self.active_start: self.active_end] += 1
        # res_matrix = tf.stop_gradient(mask_h*E) + mask*E
        self.embeddings = K.stop_gradient(self.embeddings * (1 - mask)) + self.embeddings * mask
        return [super(PartialEmbedding, self).call(inputs), self.embeddings + 0]

    def compute_output_shape(self, input_shape: Tensors) -> List[Tensors]:
        return [super(PartialEmbedding, self).compute_output_shape(input_shape), K.int_shape(self.embeddings)]


class PTuniningPrompt(BasePromptModel):

    def __init__(self, plm_backbone: str, plm_config_path: str, plm_ckpt_path: str,
                 template: Template, learning_rate: float = 0.00001,
                 freeze_plm: bool = True, encoder: str = 'mlp') -> None:
        """ PTuning Prompt Model
        Args:
          - plm_backbone: str, backbone of pretrained language model
          - plm_config_path: str, configure path of pretrained language model
          - plm_ckpt_path: str, checkpoint path of pretrained language model
          - template: List[str], template
          - label_tokens_map: str, verbalizer, map of label to tokens
          - tokenizer: langml.Tokenizer, tokenizer
          - learning_rate: float, learning rate
          - freeze_plm: bool, whether to freeze pretrained language model weights
          - encoder: str, template encoder, [`mlp`, `lstm`], default `mlp`
        """
        self.encoder = encoder.lower()
        super().__init__(
            plm_backbone, plm_config_path, plm_ckpt_path, template,
            learning_rate=learning_rate, freeze_plm=freeze_plm)

    def build_model(self) -> Models:
        template_in = L.Input(shape=(None,), name='Input-Template-Mask')
        template_mask = L.Lambda(lambda x: K.cast(
            K.greater(K.expand_dims(x, 2), 0), K.floatx()))(template_in)
        self.plm.token_embedding_layer = PartialEmbedding(
            input_dim=self.plm.vocab_size,
            output_dim=self.plm.embedding_dim,
            active_start=1,
            active_end=len(self.template) + 1,
            mask_zero=True,
            trainable=self.plm.trainable,
            embeddings_initializer=self.plm.initializer,
            name=self.plm.get_weight_name('Embedding-Token'),
        )

        def custom_embedding_callback(inputs):
            embedding, embedding_weights = self.plm.get_embedding(inputs)
            template_embedding = L.Multiply()([embedding, template_mask])
            if self.encoder == 'lstm':
                template_embedding = L.LSTM(self.plm.embedding_dim,
                                            return_sequences=True,
                                            name='Template-LSTM-Encoder')(template_embedding)
            template_embedding = L.Dense(self.plm.embedding_dim * 2,
                                         activation='relu',
                                         name='Template-Dense-Hidden')(template_embedding)
            template_embedding = L.Dense(self.plm.embedding_dim,
                                         name='Template-Dense-Output')(template_embedding)
            template_embedding = L.Multiply()([template_embedding, template_mask])
            embedding = L.Add()([embedding, template_embedding])
            return embedding, embedding_weights

        inputs = self.plm.get_inputs()
        outputs = self.plm(inputs,
                           return_model=False,
                           with_mlm=True,
                           with_nsp=False,
                           custom_embedding_callback=custom_embedding_callback)

        model = keras.Model((template_in, *inputs), outputs)
        if self.freeze_plm:
            for layer in model.layers:
                if not (layer.name.startswith('Template-') or layer.name != 'Embedding-Token'):
                    layer.trainable = False
        model.summary()
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss='sparse_categorical_crossentropy',
        )
        self.lazy_restore_callback(model)
        return model
