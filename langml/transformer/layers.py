# -*- coding: utf-8 -*-

""" Yet another transformer implementation.
"""

# TODO: Transformer Decoder

from typing import Optional, List, Union, Any

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.backend as K
    import keras.layers as L
from langml.tensor_typing import Tensors, Activation, Initializer, Constraint, Regularizer


class FeedForward(L.Layer):
    """ Feed Forward Layer
    https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self,
                 units,
                 activation: Activation = 'relu',
                 kernel_initializer: Initializer = 'glorot_normal',
                 kernel_regularizer: Optional[Regularizer] = None,
                 kernel_constraint: Optional[Constraint] = None,
                 bias_initializer: Initializer = 'zeros',
                 bias_regularizer: Optional[Regularizer] = None,
                 bias_constraint: Optional[Constraint] = None,
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

    def get_config(self) -> dict:
        config = {
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate
        }
        base_config = super(FeedForward, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name=f'{self.name}_W1',
        )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        if self.dropout_rate > 0.0:
            self.dropout_layer = L.Dropout(self.dropout_rate)
        super(FeedForward, self).build(input_shape)

    def call(self,
             inputs: Tensors,
             mask: Optional[Tensors] = None,
             training: Optional[Any] = None,
             **kwargs) -> Union[List[Tensors], Tensors]:
        hidden = K.dot(inputs, self.W1)
        if self.use_bias:
            hidden = K.bias_add(hidden, self.b1)
        if self.activation is not None:
            hidden = self.activation(hidden)
        if self.dropout_rate > 0.0:
            hidden = self.dropout_layer(hidden)
        output = K.dot(hidden, self.W2)
        if self.use_bias:
            output = K.bias_add(output, self.b2)
        return output

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Union[Tensors, List[Tensors]]] = None) -> Union[
                         List[Union[Tensors, None]], Tensors]:
        return mask

    @staticmethod
    def get_custom_objects() -> dict:
        return {'FeedForward': FeedForward}

    def compute_output_shape(self, input_shape: Tensors) -> Tensors:
        return input_shape
