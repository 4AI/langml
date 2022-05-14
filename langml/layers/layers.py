# -*- coding: utf-8 -*-

from typing import Optional

import tensorflow as tf
from langml import keras, K, L
from langml.tensor_typing import Tensors, Initializer, Constraint, Regularizer


class AbsolutePositionEmbedding(L.Layer):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 mode: str = 'add',
                 embeddings_initializer: Initializer = 'uniform',
                 embeddings_regularizer: Optional[Regularizer] = None,
                 embeddings_constraint: Optional[Constraint] = None,
                 mask_zero: bool = False,
                 **kwargs):
        """Absolute Position Embedding
        # mode:
          expand
            # Input shape
                2D tensor with shape: `(batch_size, sequence_length)`.
            # Output shape
                3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
          add
            # Input shape
                3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
            # Output shape
                3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
          concat
            # Input shape
                3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
            # Output shape
                3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
        """
        assert mode in ['expand', 'add', 'concat'], f'not support mode `{mode}`, options: expand | add | concat'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = True if mask_zero else False
        self.embeddings = None
        super(AbsolutePositionEmbedding, self).__init__(**kwargs)

    def get_config(self) -> dict:
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'mode': self.mode,
            'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint),
            'mask_zero': self.mask_zero
        }
        base_config = super(AbsolutePositionEmbedding, self).get_config()
        return dict(base_config, **config)

    @staticmethod
    def get_custom_objects() -> dict:
        return {'AbsolutePositionEmbedding': AbsolutePositionEmbedding}

    def build(self, input_shape: Tensors):
        if self.mode == 'expand':
            self.embeddings = self.add_weight(
                shape=(self.input_dim * 2 + 1, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        super(AbsolutePositionEmbedding, self).build(input_shape)

    def compute_mask(self, inputs: Tensors, mask: Optional[Tensors] = None) -> Tensors:
        if self.mode == 'expand':
            if self.mask_zero:
                output_mask = K.not_equal(inputs, self.mask_zero)
            else:
                output_mask = None
        else:
            output_mask = mask
        return output_mask

    def compute_output_shape(self, input_shape: Tensors) -> Tensors:
        if self.mode == 'expand':
            return input_shape + (self.output_dim,)
        if self.mode == 'concat':
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs: Tensors, **kwargs) -> Tensors:
        if self.mode == 'expand':
            inputs = K.cast(inputs, 'int32')
            return K.gather(
                self.embeddings,
                K.minimum(K.maximum(inputs, -self.input_dim), self.input_dim) + self.input_dim,
            )
        input_shape = K.shape(inputs)
        if self.mode == 'add':
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
        pos_embeddings = K.tile(
            K.expand_dims(self.embeddings[:seq_len, :output_dim], axis=0),
            [batch_size, 1, 1],
        )
        if self.mode == 'add':
            return inputs + pos_embeddings
        return K.concatenate([inputs, pos_embeddings], axis=-1)


class SinusoidalPositionEmbedding(L.Layer):
    """ Sinusoidal Position Embedding
    """
    def __init__(self,
                 output_dim: int,
                 mode: str = 'add',
                 mask_zero: bool = False,
                 **kwargs):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        assert mode in ['add', 'mul', 'zero'], f'not support mode `{mode}`, options: add | mul | zero'
        self.output_dim = output_dim
        self.mode = mode
        self.supports_masking = mask_zero
        self.mask_zero = mask_zero

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "mode": self.mode,
            "mask_zero": self.mask_zero
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(base_config, **config)

    def call(self, inputs: Tensors) -> Tensors:
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.mode == 'add':
            return inputs + embeddings
        elif self.mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.mode == 'zero':
            return embeddings

        embeddings = K.tile(embeddings, [batch_size, 1, 1])
        return K.concatenate([inputs, embeddings])

    def compute_mask(self, inputs: Tensors, mask: Optional[Tensors] = None) -> Tensors:
        if self.mask_zero:
            mask = K.not_equal(inputs, self.mask_zero)
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.output_dim,)

    @staticmethod
    def get_custom_objects() -> dict:
        return {'SinusoidalPositionEmbedding': SinusoidalPositionEmbedding}


class ScaleOffset(L.Layer):
    """ Scale Offset
    """
    def __init__(self, scale: bool = True, offset: bool = True, **kwargs):
        super(ScaleOffset, self).__init__(**kwargs)
        self.scale = scale
        self.offset = offset

        self.supports_masking = True

    def get_config(self):
        config = {
            'scale': self.scale,
            'offset': self.offset,
        }
        base_config = super(ScaleOffset, self).get_config()
        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        super(ScaleOffset, self).build(input_shape)

        if self.offset is True:
            self.beta = self.add_weight(
                name='beta', shape=(input_shape[-1],), initializer='zeros'
            )
        if self.scale is True:
            self.gamma = self.add_weight(
                name='gamma', shape=(input_shape[-1],), initializer='ones'
            )

    def compute_mask(self, inputs: Tensors, mask: Optional[Tensors] = None):
        return mask

    def call(self, inputs: Tensors) -> Tensors:
        o = inputs
        if self.scale:
            o *= self.gamma
        if self.offset:
            o += self.beta
        return o

    def compute_output_shape(self, input_shape: Tensors) -> Tensors:
        return input_shape

    @staticmethod
    def get_custom_objects() -> dict:
        return {'ScaleOffset': ScaleOffset}