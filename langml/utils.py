# -*- coding: utf-8 -*-

import functools
from typing import List, Tuple, Callable

import tensorflow as tf

from langml.log import warn
from langml.tokenizer import Tokenizer, WPTokenizer, SPTokenizer
from langml import TF_KERAS
if TF_KERAS:
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # NOQA
else:
    from keras.preprocessing.sequence import pad_sequences  # NOQA


def deprecated_warning(msg='this function is deprecated! it might be removed in a future version.'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn(msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def bio_decode(tags: List[str]) -> List[Tuple[int, int, str]]:
    """ Decode BIO tags

    Examples:
    >>> bio_decode(['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'I-ORG'])
    >>> [(0, 1, 'PER'), (3, 5, 'ORG')]
    """
    entities = []
    start_tag = None
    for i, tag in enumerate(tags):
        tag_capital = tag.split('-')[0]
        tag_name = tag.split('-')[1] if tag != 'O' else ''
        if tag_capital in ['B', 'O']:
            if start_tag is not None:
                entities.append((start_tag[0], i - 1, start_tag[1]))
                start_tag = None
            if tag_capital == 'B':
                start_tag = (i, tag_name)
        elif tag_capital == 'I' and start_tag is not None and start_tag[1] != tag_name:
            entities.append((start_tag[0], i, start_tag[1]))
            start_tag = None
    if start_tag is not None:
        entities.append((start_tag[0], i, start_tag[1]))
    return entities


def load_variables(checkpoint_path: str) -> Callable:
    """ load variables from chechkpoint
    """
    def wrap(varname: str):
        return tf.train.load_variable(checkpoint_path, varname)
    return wrap


def auto_tokenizer(vocab_path: str, lowercase: bool = False) -> Tokenizer:
    if vocab_path.endswith('.txt'):
        tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
    elif vocab_path.endswith('.model'):
        tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
    else:
        raise ValueError("Langml cannot deduce which tokenizer to apply")  # NOQA
    return tokenizer
