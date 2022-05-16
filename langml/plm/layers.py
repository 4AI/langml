# -*- coding: utf-8 -*-

from typing import Optional, List, Union

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


class TokenEmbedding(L.Embedding):
    @staticmethod
    def get_custom_objects() -> dict:
        return {'TokenEmbedding': TokenEmbedding}

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> List[Union[Tensors, None]]:
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs: Tensors) -> List[Tensors]:
        return [super(TokenEmbedding, self).call(inputs), self.embeddings + 0]

    def compute_output_shape(self, input_shape: Tensors) -> List[Tensors]:
        return [super(TokenEmbedding, self).compute_output_shape(input_shape), K.int_shape(self.embeddings)]


class EmbeddingMatching(L.Layer):
    def __init__(self,
                 initializer: Initializer = 'zeros',
                 regularizer: Optional[Regularizer] = None,
                 constraint: Optional[Constraint] = None,
                 use_bias: bool = True,
                 **kwargs):
        super(EmbeddingMatching, self).__init__(**kwargs)
        self.supports_masking = True
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.use_bias = use_bias

    def get_config(self) -> dict:
        config = {
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
        }
        base_config = super(EmbeddingMatching, self).get_config()
        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(int(input_shape[1][0]), ),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
                name='bias',
            )
        super(EmbeddingMatching, self).build(input_shape)

    def compute_mask(self, inputs: Tensors, mask: Optional[Tensors] = None) -> Tensors:
        if isinstance(mask, list):
            return mask[0]
        return mask

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Tensors:
        inputs, embeddings = inputs
        output = K.dot(inputs, K.transpose(embeddings))
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        return K.softmax(output)

    @staticmethod
    def get_custom_objects() -> dict:
        return {'EmbeddingMatching': EmbeddingMatching}

    def compute_output_shape(self, input_shape: Tensors) -> Tensors:
        return input_shape[0][:2] + (input_shape[1][0], )


class Masked(L.Layer):
    """Generate output mask based on the given mask.
    https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self,
                 return_masked: bool = False,
                 **kwargs):
        super(Masked, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_masked = return_masked

    @staticmethod
    def get_custom_objects() -> dict:
        return {'Masked': Masked}

    def get_config(self) -> dict:
        config = {
            'return_masked': self.return_masked,
        }
        base_config = super(Masked, self).get_config()
        return dict(base_config, **config)

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[List[Union[Tensors, None]], Tensors]:
        token_mask = K.not_equal(inputs[1], 0)
        masked = K.all(K.stack([token_mask, mask[0]], axis=0), axis=0)
        if self.return_masked:
            return [masked, None]
        return masked

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Tensors:
        output = inputs[0] + 0
        if self.return_masked:
            return [output, K.cast(self.compute_mask(inputs, mask)[0], K.floatx())]
        return output

    def compute_output_shape(self, input_shape: Tensors) -> Union[List[Tensors], Tensors]:
        if self.return_masked:
            return [input_shape[0], (2, ) + input_shape[1]]
        return input_shape[0]
