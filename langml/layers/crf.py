# -*- coding: utf-8 -*-

from typing import Optional, Callable, Union

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras.backend as K
    import keras.layers as L

import tensorflow as tf

from langml.tensor_typing import Tensors
from langml.third_party.crf import crf_log_likelihood, crf_decode


class CRF(L.Layer):
    def __init__(self,
                 output_dim: int,
                 sparse_target: bool = True,
                 **kwargs):
        """
        Args:
            output_dim (int): the number of labels to tag each temporal input.
            sparse_target (bool): whether the the ground-truth label represented in one-hot.
        Input shape:
            (batch_size, sentence length, output_dim)
        Output shape:
            (batch_size, sentence length, output_dim)

        Usage:
        >>> from tensorflow.keras.models import Sequential
        >>> from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense


        >>> num_labels = 10
        >>> embedding_size = 100
        >>> hidden_size = 128

        >>> model = Sequential()
        >>> model.add(Embedding(num_labels, embedding_size))
        >>> model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        >>> model.add(Dense(num_labels))

        >>> crf = CRF(num_labels, sparse_target=True)
        >>> model.add(crf)
        >>> model.compile('adam', loss=crf.loss, metrics=[crf.accuracy])
        """
        super(CRF, self).__init__(**kwargs)
        self.support_mask = True
        self.output_dim = output_dim
        self.sparse_target = sparse_target
        self.input_spec = L.InputSpec(min_ndim=3)
        self.supports_masking = False
        self.sequence_lengths = None

    def build(self, input_shape: Tensors):
        assert len(input_shape) == 3
        f_shape = input_shape
        input_spec = L.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None):
        return None

    def call(self,
             inputs: Tensors,
             sequence_lengths: Optional[Tensors] = None,
             training: Optional[Union[bool, int]] = None,
             mask: Optional[Tensors] = None,
             **kwargs) -> Tensors:
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tf.convert_to_tensor(sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            self.sequence_lengths = K.flatten(sequence_lengths)
        else:
            self.sequence_lengths = tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * (
                tf.shape(inputs)[1]
            )

        viterbi_sequence, _ = crf_decode(sequences,
                                         self.transitions,
                                         self.sequence_lengths)
        output = K.one_hot(viterbi_sequence, self.output_dim)
        return K.in_train_phase(sequences, output, training=training)

    @property
    def loss(self) -> Callable:
        def crf_loss(y_true: Tensors, y_pred: Tensors) -> Tensors:
            y_true = K.argmax(y_true, 2) if self.sparse_target else y_true
            y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
            log_likelihood, _ = crf_log_likelihood(
                y_pred,
                K.cast(y_true, dtype='int32'),
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            return K.mean(-log_likelihood)
        return crf_loss

    @property
    def accuracy(self) -> Callable:
        def viterbi_accuracy(y_true: Tensors, y_pred: Tensors) -> Tensors:
            y_true = K.argmax(y_true, 2) if self.sparse_target else y_true
            y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
            viterbi_sequence, _ = crf_decode(
                y_pred,
                self.transitions,
                self.sequence_lengths
            )
            mask = K.all(K.greater(y_pred, -1e6), axis=2)
            mask = K.cast(mask, K.floatx())
            y_true = K.cast(y_true, 'int32')
            corrects = K.cast(K.equal(y_true, viterbi_sequence), K.floatx())
            return K.sum(corrects * mask) / K.sum(mask)
        return viterbi_accuracy

    def compute_output_shape(self, input_shape: Tensors) -> Tensors:
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    @property
    def trans(self) -> Tensors:
        """ transition parameters
        """
        return K.eval(self.transitions)

    def get_config(self) -> dict:
        config = {
            'output_dim': self.output_dim,
            'sparse_target': self.sparse_target,
            'supports_masking': self.supports_masking,
            'transitions': K.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(base_config, **config)

    @staticmethod
    def get_custom_objects() -> dict:
        return {'CRF': CRF}
