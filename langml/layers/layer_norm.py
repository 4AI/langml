# -*- coding: utf-8 -*-

from typing import Optional, Union

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.backend as K
    import keras.layers as L

from langml.tensor_typing import Tensors, Initializer, Constraint, Regularizer


class LayerNorm(L.Layer):
    def __init__(self,
                 center: bool = True,
                 scale: bool = True,
                 epsilon: float = 1e-7,
                 gamma_initializer: Initializer = 'ones',
                 gamma_regularizer: Optional[Regularizer] = None,
                 gamma_constraint: Optional[Constraint] = None,
                 beta_initializer: Initializer = 'zeros',
                 beta_regularizer: Optional[Regularizer] = None,
                 beta_constraint: Optional[Constraint] = None,
                 **kwargs):
        super(LayerNorm, self).__init__(**kwargs)

        self.supports_masking = True

        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)

    def get_config(self) -> dict:
        config = {
            "center": self.center,
            "scale": self.scale,
            "epsilon": self.epsilon,
            "gamma_initializer": keras.initializers.serialize(self.gamma_initializer),
            "gamma_regularizer": keras.regularizers.serialize(self.gamma_regularizer),
            "gamma_constraint": keras.constraints.serialize(self.gamma_constraint),
            "beta_initializer": keras.initializers.serialize(self.beta_initializer),
            "beta_regularizer": keras.regularizers.serialize(self.beta_regularizer),
            "beta_constraint": keras.constraints.serialize(self.beta_constraint)
        }
        base_config = super(LayerNorm, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs: Tensors, **kwargs) -> Tensors:
        # layer norm: specify axis=-1
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        # standard normalization: x = (x - \mu) / \std
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[Tensors, None]:
        return mask

    @staticmethod
    def get_custom_objects() -> dict:
        return {'LayerNorm': LayerNorm}

    def compute_output_shape(self, input_shape: Tensors) -> Tensors:
        return input_shape
