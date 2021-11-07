# -*- coding: utf-8 -*-

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
else:
    import keras.backend as K
    import keras.layers as L

import numpy as np
import pytest


@pytest.mark.parametrize("test_input,expected", [
    (K.constant([[1., 2., 3.],
                [4., 5., 6.]]), 
     np.array([[-0.46291006, 0., 0.46291006],
               [-0.19738552, 0., 0.19738552]])),
    (K.constant([[[1., 2., 3.]],
                [[4., 5., 6.]]]),
     np.array([[[-0.46291006, 0., 0.46291006]],
               [[-0.19738552, 0., 0.19738552]]]))
])
def test_layer_norm(test_input, expected):
    from langml.layers import LayerNorm

    output = LayerNorm(name='layer_norm')(test_input)
    assert np.allclose(expected, np.array(K.eval(output).tolist()))
