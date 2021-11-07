# -*- coding: utf-8 -*-

import os
import random
import string
from typing import Any

import tensorflow as tf
from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras.backend as K
else:
    import keras.backend as K

from langml.tensor_typing import Models
from langml.log import info, warn
from langml import TF_VERSION


SAVED_MODEL_TAG = 'serve'


def get_random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def export_model_v1(model, export_model_dir):
    """
    :param export_model_dir: type string, save dir for exported model url
    :param model_version: type int best
    :return:no return
    """

    if os.path.exists(export_model_dir):
        warn(f'path `{export_model_dir}` exists!')
        export_model_dir = f"{export_model_dir}.{get_random_string(6)}"
        warn(f'auto relocation to `{export_model_dir}`')

    os.makedirs(export_model_dir)

    with tf.get_default_graph().as_default():
        info(f"input: {model.input}")
        info(f"output: {model.output}")
        input_map = {}
        if isinstance(model.input, (tuple, list)):
            for x in model.input:
                input_map[x.name.split(':')[0]] = tf.saved_model.build_tensor_info(x)
        else:
            input_map[model.input.name.split(':')[0]] = tf.saved_model.build_tensor_info(model.input)
        info(f'input map: {input_map}')
        output_map = {}
        if isinstance(model.output, (tuple, list)):
            for x in model.output:
                output_map[x.name.split(':')[0]] = tf.saved_model.build_tensor_info(x)
        else:
            output_map[model.output.name.split(':')[0]] = tf.saved_model.build_tensor_info(model.output)
        info(f'output map: {output_map}')
        prediction_signature = (
            tf.saved_model.build_signature_def(
                inputs=input_map,
                outputs=output_map)
        )
        info('step1 => prediction_signature created successfully')
        builder = tf.saved_model.builder.SavedModelBuilder(export_model_dir)
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[SAVED_MODEL_TAG],
            signature_def_map={
                'predict': prediction_signature,
                'serving_default': prediction_signature,
            },
        )
        info(f'step2 => Export path({export_model_dir}) ready to export trained model')
        builder.save()
        info(f'done! model has saved to {export_model_dir}.')


def save_frozen(model: Models, fpath: str):
    if int(tf.__version__.split('.')[0]) > 1:
        tf.saved_model.save(model, fpath)
    else:
        info('apply tensorflow 1.x frozen')
        export_model_v1(model, fpath)


def load_frozen(model_dir: str,
                session: Any = None) -> Any:

    if TF_VERSION > 1:
        return tf.saved_model.load(model_dir)

    if session is None:
        raise ValueError('session is required in tensorflow 1.x')
    tf.saved_model.loader.load(session, [SAVED_MODEL_TAG], export_dir=model_dir)
    info('done! session has restored.')
    return session
