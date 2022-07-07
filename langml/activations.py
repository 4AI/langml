# -*- coding: utf-8 -*-

""" Activations
"""

import math

from langml import keras, K, L
from langml.tensor_typing import Tensors


def gelu(x: Tensors) -> Tensors:
    r""" Gaussian Error Linear Units (GELUs)
    https://arxiv.org/abs/1606.08415

    $GELU(x) = 0.5x(1 + tanh[\sqrt(2 / \Pi) (x + 0.044715x^3)])$
    """
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def relu2(x: Tensors) -> Tensors:
    """ ReLU Square
    """
    return K.pow(K.relu(x), 2)


def swish(x: Tensors, beta: float = 1.0) -> Tensors:
    return (x * K.sigmoid(beta * x))


custom_objects = {}
custom_objects.update({'gelu': L.Activation(gelu)})
custom_objects.update({'relu2': L.Activation(relu2)})
custom_objects.update({'swish': L.Activation(swish)})

keras.utils.get_custom_objects().update(custom_objects)
