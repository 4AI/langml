# -*- coding: utf-8 -*-

from typing import Optional, Union, List

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.backend as K
    import keras.layers as L

from langml.activations import relu2
from langml.layers import SineCosinePositionEmbedding, ScaleOffset
from langml.tensor_typing import Tensors, Activation, Initializer, Constraint, Regularizer


class SelfAttention(L.Layer):
    def __init__(self,
                 attention_units: Optional[int] = None,
                 return_attention: bool = False,
                 is_residual: bool = False,
                 attention_activation: Activation = 'relu',
                 attention_epsilon: float = 1e10,
                 kernel_initializer: Initializer = 'glorot_normal',
                 kernel_regularizer: Optional[Regularizer] = None,
                 kernel_constraint: Optional[Constraint] = None,
                 bias_initializer: Initializer = 'zeros',
                 bias_regularizer: Optional[Regularizer] = None,
                 bias_constraint: Optional[Constraint] = None,
                 use_attention_bias: bool = True,
                 attention_penalty_weight: float = 0.0,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self.supports_masking = True

        self.attention_units = attention_units
        self.return_attention = return_attention
        self.is_residual = is_residual
        self.attention_epsilon = attention_epsilon
        self.attention_activation = keras.activations.get(attention_activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.use_attention_bias = use_attention_bias
        self.attention_penalty_weight = attention_penalty_weight

    def get_config(self) -> dict:
        config = {
            "attention_units": self.attention_units,
            "return_attention": self.return_attention,
            "is_residual": self.is_residual,
            "attention_epsilon": self.attention_epsilon,
            "attention_activation": keras.activations.serialize(self.attention_activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "use_attention_bias": self.use_attention_bias,
            "attention_penalty_weight": self.attention_penalty_weight
        }
        base_config = super(SelfAttention, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        feature_dim = int(input_shape[2])
        units = feature_dim if self.attention_units is None else self.attention_units

        self.Wq = self.add_weight(shape=(feature_dim, units),
                                  name=f'{self.name}_Attn_Wq',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wk = self.add_weight(shape=(feature_dim, units),
                                  name=f'{self.name}_Attn_Wt',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wv = self.add_weight(shape=(feature_dim, units),
                                  name=f'{self.name}_Attn_Wv',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.attn_bias = self.add_weight(shape=(1,),
                                             name=f'{self.name}_Attn_bias',
                                             initializer=self.bias_initializer,
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Union[List[Tensors], Tensors]:

        q = K.dot(inputs, self.Wq)
        k = K.dot(inputs, self.Wk)
        v = K.dot(inputs, self.Wv)
        if self.attention_activation is not None:
            q = self.attention_activation(q)
            k = self.attention_activation(k)
            v = self.attention_activation(v)

        if self.use_attention_bias:
            q += self.attn_bias
            k += self.attn_bias
            v += self.attn_bias

        e = K.batch_dot(q, k, axes=2)

        if mask is not None:
            if len(K.int_shape(mask)) == len(K.int_shape(inputs)) - 1:
                mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= self.attention_epsilon * (1.0 - mask)

        a = K.softmax(e)
        v_o = K.batch_dot(a, v)

        if self.is_residual:
            v_o += v

        if self.attention_penalty_weight > 0.0:
            self.add_loss(self._attention_penalty(a))

        if self.return_attention:
            return [v_o, a]

        return v_o

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[List[Union[Tensors, None]], Tensors]:
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_penalty(self, attention: Tensors) -> Tensors:
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_penalty_weight * K.sum(K.square(K.batch_dot(
            attention, K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects() -> dict:
        return {'SelfAttention': SelfAttention}

    def compute_output_shape(self, input_shape: Tensors) -> Union[List[Tensors], Tensors]:
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape


class SelfAdditiveAttention(L.Layer):
    def __init__(self,
                 attention_units: Optional[int] = None,
                 return_attention: bool = False,
                 is_residual: bool = False,
                 attention_activation: Activation = 'relu',
                 attention_epsilon: float = 1e10,
                 kernel_initializer: Initializer = 'glorot_normal',
                 kernel_regularizer: Optional[Regularizer] = None,
                 kernel_constraint: Optional[Constraint] = None,
                 bias_initializer: Initializer = 'zeros',
                 bias_regularizer: Optional[Regularizer] = None,
                 bias_constraint: Optional[Constraint] = None,
                 use_attention_bias: bool = True,
                 attention_penalty_weight: float = 0.0,
                 **kwargs):
        super(SelfAdditiveAttention, self).__init__(**kwargs)

        self.supports_masking = True

        self.attention_units = attention_units
        self.return_attention = return_attention
        self.is_residual = is_residual
        self.attention_epsilon = attention_epsilon
        self.attention_activation = keras.activations.get(attention_activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.use_attention_bias = use_attention_bias
        self.attention_penalty_weight = attention_penalty_weight

    def get_config(self) -> dict:
        config = {
            "attention_units": self.attention_units,
            "return_attention": self.return_attention,
            "is_residual": self.is_residual,
            "attention_epsilon": self.attention_epsilon,
            "attention_activation": keras.activations.serialize(self.attention_activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "use_attention_bias": self.use_attention_bias,
            "attention_penalty_weight": self.attention_penalty_weight
        }
        base_config = super(SelfAdditiveAttention, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        feature_dim = int(input_shape[2])
        units = feature_dim if self.attention_units is None else self.attention_units

        self.Wh = self.add_weight(shape=(feature_dim, units),
                                  name=f'{self.name}_Attn_Wh',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.We = self.add_weight(shape=(units, 1),
                                  name=f'{self.name}_Attn_We',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.attn_bias = self.add_weight(shape=(1,),
                                             name=f'{self.name}_Attn_bias',
                                             initializer=self.bias_initializer,
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Union[List[Tensors], Tensors]:
        h = K.dot(inputs, self.Wh)
        if self.attention_activation is not None:
            h = self.attention_activation(h)
        if self.use_attention_bias:
            h += self.attn_bias
        e = K.dot(h, self.We)
        if self.use_attention_bias:
            e += self.attn_bias

        if mask is not None:
            if len(K.int_shape(mask)) == len(K.int_shape(inputs)) - 1:
                mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= self.attention_epsilon * (1.0 - mask)

        a = K.softmax(e, axis=1)
        v_o = a * inputs

        if self.is_residual:
            v_o += inputs

        if self.attention_penalty_weight > 0.0:
            self.add_loss(self._attention_penalty(a))

        if self.return_attention:
            return [v_o, a]

        return v_o

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[List[Union[Tensors, None]], Tensors]:
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_penalty(self, attention: Tensors) -> Tensors:
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_penalty_weight * K.sum(K.square(K.batch_dot(
            attention, K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects() -> dict:
        return {'SelfAdditiveAttention': SelfAdditiveAttention}

    def compute_output_shape(self, input_shape: Tensors) -> Union[List[Tensors], Tensors]:
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape


class ScaledDotProductAttention(L.Layer):
    r""" ScaledDotProductAttention

    $Attention(Q, K, V) = softmax(\frac{Q K^T}{\sqrt{d_k}}) V$

    https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self,
                 return_attention: bool = False,
                 history_only: bool = False,
                 **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

        self.supports_masking = True

        self.return_attention = return_attention
        self.history_only = history_only

    def get_config(self) -> dict:
        config = {
            "return_attention": self.return_attention,
            "history_only": self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()

        return dict(base_config, **config)

    def call(self,
             inputs: Tensors,
             mask: Optional[Union[Tensors, List[Tensors]]] = None, **kwargs) -> Union[List[Tensors], Tensors]:
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            mask = mask[1]
        # e = \frac{QK^T}{\sqrt{d_k}}
        # shape: [(B, Lq, D), (B, Lk, D)] -> (B, Lq, Lk)
        e = K.batch_dot(q, k, axes=2) / K.sqrt(K.cast(K.shape(q)[-1], dtype=K.floatx()))
        if self.history_only:
            q_len, k_len = K.shape(q)[1], K.shape(k)[1]
            indices = K.expand_dims(K.arange(0, k_len), axis=0)
            upper = K.expand_dims(K.arange(0, q_len), axis=-1)
            e -= 10000.0 * K.expand_dims(K.cast(indices > upper, K.floatx()), axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx()))
        # softmax(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        attention = e / K.sum(e, axis=-1, keepdims=True)
        v = K.batch_dot(attention, v)
        if self.return_attention:
            return [v, attention]
        return v

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Union[Tensors, List[Tensors]]] = None) -> Union[
                         List[Union[Tensors, None]], Tensors]:
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    @staticmethod
    def get_custom_objects() -> dict:
        return {'ScaledDotProductAttention': ScaledDotProductAttention}

    def compute_output_shape(self, input_shape: Union[Tensors, List[Tensors]]) -> Union[List[Tensors], Tensors]:
        if isinstance(input_shape, list):
            q_shape, k_shape, v_shape = input_shape
        else:
            q_shape = k_shape = v_shape = input_shape
        output_shape = q_shape[:-1] + v_shape[-1:]
        if self.return_attention:
            attention_shape = q_shape[:2] + (k_shape[1],)
            return [output_shape, attention_shape]
        return output_shape


class MultiHeadAttention(L.Layer):
    """ MultiHeadAttention
    https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self,
                 head_num: int,
                 return_attention: bool = False,
                 attention_activation: Activation = 'relu',
                 kernel_initializer: Initializer = 'glorot_normal',
                 kernel_regularizer: Optional[Regularizer] = None,
                 kernel_constraint: Optional[Constraint] = None,
                 bias_initializer: Initializer = 'zeros',
                 bias_regularizer: Optional[Regularizer] = None,
                 bias_constraint: Optional[Constraint] = None,
                 use_attention_bias: bool = True,
                 history_only: bool = False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.supports_masking = True

        self.head_num = head_num
        self.return_attention = return_attention
        self.attention_activation = keras.activations.get(attention_activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.use_attention_bias = use_attention_bias
        self.history_only = history_only

    def get_config(self) -> dict:
        config = {
            "head_num": self.head_num,
            "return_attention": self.return_attention,
            "attention_activation": keras.activations.serialize(self.attention_activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "use_attention_bias": self.use_attention_bias,
            "history_only": self.history_only
        }
        base_config = super(MultiHeadAttention, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        assert feature_dim % self.head_num == 0, 'feature_dim should be divided by head_num with no remainder'
        self.Wq = self.add_weight(shape=(int(q[-1]), feature_dim),
                                  name=f'{self.name}_Wq',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wk = self.add_weight(shape=(int(k[-1]), feature_dim),
                                  name=f'{self.name}_Wk',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wv = self.add_weight(shape=(feature_dim, feature_dim),
                                  name=f'{self.name}_Wv',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wo = self.add_weight(shape=(feature_dim, feature_dim),
                                  name=f'{self.name}_Wo',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.bq = self.add_weight(shape=(feature_dim,),
                                      name=f'{self.name}_bq',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
            self.bk = self.add_weight(shape=(feature_dim,),
                                      name=f'{self.name}_bk',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
            self.bv = self.add_weight(shape=(feature_dim,),
                                      name=f'{self.name}_bv',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
            self.bo = self.add_weight(shape=(feature_dim,),
                                      name=f'{self.name}_bo',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        return K.permute_dimensions(x, [0, 2, 1, 3])

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Tensors:
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_attention_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.attention_activation is not None:
            q = self.attention_activation(q)
            k = self.attention_activation(k)
            v = self.attention_activation(v)
        scaled_dot_product_attention = ScaledDotProductAttention(
            return_attention=True,
            history_only=self.history_only,
            name=f'{self.name}-Attention',
        )
        output, attention = scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        attention = self._reshape_attention_from_batches(attention, self.head_num)
        output = self._reshape_from_batches(output, self.head_num)
        output = K.dot(output, self.Wo)
        if self.use_attention_bias:
            output += self.bo
        if self.attention_activation is not None:
            output = self.attention_activation(output)
        if self.return_attention:
            return [output, attention]
        return output

    @staticmethod
    def get_custom_objects() -> dict:
        return {'MultiHeadAttention': MultiHeadAttention}

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[List[Union[Tensors, None]], Tensors]:
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    def compute_output_shape(self, input_shape: Union[Tensors, List[Tensors]]) -> Union[List[Tensors], Tensors]:
        if isinstance(input_shape, list):
            q_shape, _, v_shape = input_shape
        else:
            q_shape = _ = v_shape = input_shape
        output_shape = q_shape[:-1] + (v_shape[-1],)
        if self.return_attention:
            attention_shape = (*q_shape[:-1], self.head_num, v_shape[-1])
            return [output_shape, attention_shape]
        return output_shape


class GatedAttentionUnit(L.Layer):
    """ Gated Attention Unit
    https://arxiv.org/abs/2202.10447
    """
    def __init__(self,
                 attention_units: int,
                 attention_activation: Activation = 'relu',
                 attention_normalizer: Activation = relu2,
                 attention_epsilon: float = 1e10,
                 kernel_initializer: Initializer = 'glorot_normal',
                 kernel_regularizer: Optional[Regularizer] = None,
                 kernel_constraint: Optional[Constraint] = None,
                 bias_initializer: Initializer = 'zeros',
                 bias_regularizer: Optional[Regularizer] = None,
                 bias_constraint: Optional[Constraint] = None,
                 use_attention_bias: bool = True,
                 use_attention_scale: bool = True,
                 use_relative_position: bool = True,
                 use_offset: bool = True,
                 use_scale: bool = True,
                 **kwargs):
        super(GatedAttentionUnit, self).__init__(**kwargs)

        self.supports_masking = True

        self.attention_units = attention_units
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_normalizer = (
            keras.activations.get(attention_normalizer)
            if isinstance(attention_normalizer, str)
            else attention_normalizer)
        self.attention_epsilon = attention_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.use_attention_bias = use_attention_bias
        self.use_attention_scale = use_attention_scale
        self.use_relative_position = use_relative_position
        self.use_offset = use_offset
        self.use_scale = use_scale

    def get_config(self) -> dict:
        config = {
            "attention_units": self.attention_units,
            "attention_activation": keras.activations.serialize(self.attention_activation),
            "attention_normalizer": keras.activations.serialize(self.attention_normalizer),
            "attention_epsilon": self.attention_epsilon,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "use_attention_bias": self.use_attention_bias,
            "use_attention_scale": self.use_attention_scale,
            "use_relative_position": self.use_relative_position,
            "use_offset": self.use_offset,
            "use_scale": self.use_scale
        }
        base_config = super(GatedAttentionUnit, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        super(GatedAttentionUnit, self).build(input_shape)

        feature_dim = int(input_shape[-1])
        self.Wu = self.add_weight(shape=(feature_dim, 2*feature_dim),
                                  name=f'{self.name}_Wu',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wv = self.add_weight(shape=(feature_dim, 2*feature_dim),
                                  name=f'{self.name}_Wv',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wz = self.add_weight(shape=(feature_dim, self.attention_units),
                                  name=f'{self.name}_Wz',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wo = self.add_weight(shape=(2*feature_dim, feature_dim),
                                  name=f'{self.name}_Wo',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.bu = self.add_weight(shape=(2*feature_dim,),
                                      name=f'{self.name}_bu',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
            self.bv = self.add_weight(shape=(2*feature_dim,),
                                      name=f'{self.name}_bv',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
            self.bz = self.add_weight(shape=(self.attention_units,),
                                      name=f'{self.name}_bz',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
            self.bo = self.add_weight(shape=(feature_dim,),
                                      name=f'{self.name}_bo',
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.scale_offset_q = ScaleOffset(scale=self.use_scale, offset=self.use_offset,
                                          name=f'{self.name}_scale_offset_q')
        self.scale_offset_k = ScaleOffset(scale=self.use_scale, offset=self.use_offset,
                                          name=f'{self.name}_scale_offset_k')

    def apply_rotary_position_embeddings(self, sinusoidal: Tensors, *tensors):
        """ apply RoPE
        modified from: https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py#L310
        """
        def align(tensor, axes, ndim=None):
            assert len(axes) == K.ndim(tensor)
            assert ndim or min(axes) >= 0
            ndim = ndim or max(axes) + 1
            indices = [None] * ndim
            for i in axes:
                indices[i] = slice(None)
            return tensor[indices]

        assert len(tensors) > 0, 'at least one input tensor'
        assert all([
            K.int_shape(tensor) == K.int_shape(tensors[0]) for tensor in tensors[1:]
        ]), 'all tensors must have the same shape'
        ndim = K.ndim(tensors[0])
        sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
        cos_pos = K.repeat_elements(sinusoidal[..., 1::2], 2, -1)
        sin_pos = K.repeat_elements(sinusoidal[..., ::2], 2, -1)
        outputs = []
        for tensor in tensors:
            tensor2 = K.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
            tensor2 = K.reshape(tensor2, K.shape(tensor))
            outputs.append(tensor * cos_pos + tensor2 * sin_pos)
        return outputs[0] if len(outputs) == 1 else outputs

    def attn(self, x: Tensors, v: Tensors, mask: Optional[Tensors] = None) -> Tensors:
        z = K.dot(x, self.Wz)
        if self.use_attention_bias:
            z += self.bz
        z = self.attention_activation(z)
        q, k = self.scale_offset_q(z), self.scale_offset_k(z)
        if self.use_relative_position:
            pos = SineCosinePositionEmbedding("zero", output_dim=self.attention_units)(x)
            q, k = self.apply_rotary_position_embeddings(pos, q, k)
        qk = K.batch_dot(q, k, axes=2)  # (B, N, S) * (B, M, S) -> (B, N, M)
        if self.use_attention_scale:
            qk /= self.attention_units**0.5
        if mask is not None:
            if len(K.int_shape(mask)) == len(K.int_shape(x)) - 1:
                mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            qk -= self.attention_epsilon * (1.0 - mask)
        a = self.attention_normalizer(qk)
        return K.batch_dot(a, v)  # (B, N, M) * (B, M, E) -> (B, N, E)

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Tensors:
        u = K.dot(inputs, self.Wu)
        v = K.dot(inputs, self.Wv)
        if self.use_attention_bias:
            u += self.bu
            v += self.bv
        u = self.attention_activation(u)
        v = self.attention_activation(v)
        x = u * self.attn(inputs, v, mask)
        o = K.dot(x, self.Wo)
        if self.use_attention_bias:
            o += self.bo
        return inputs + o  # residual

    def compute_mask(self, inputs: Tensors, mask: Optional[Tensors] = None) -> Tensors:
        return mask

    def compute_output_shape(self, input_shape: Tensors) -> Tensors:
        return input_shape

    @staticmethod
    def get_custom_objects() -> dict:
        return {'GatedAttentionUnit': GatedAttentionUnit}
