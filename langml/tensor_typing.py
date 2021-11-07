# -*- coding: utf-8 -*-

from typing import Union, Callable, List

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
else:
    import keras

import numpy as np
import tensorflow as tf


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable, keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable, keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, keras.constraints.Constraint]
Activation = Union[None, str, Callable]
Optimizer = Union[keras.optimizers.Optimizer, str]

try:
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor

    Tensors = Union[
        List[Union[Number, list]],
        tuple,
        Number,
        np.ndarray,
        tf.Tensor,
        tf.SparseTensor,
        tf.Variable,
        KerasTensor
    ]
except ImportError:
    Tensors = Union[
        List[Union[Number, list]],
        tuple,
        Number,
        np.ndarray,
        tf.Tensor,
        tf.SparseTensor,
        tf.Variable,
    ]

Models = keras.models.Model
