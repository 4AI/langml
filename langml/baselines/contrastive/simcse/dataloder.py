# -*- coding: utf-8 -*-

import json
import math
from random import shuffle
from typing import Callable, List, Tuple

import numpy as np
import tensorflow as tf
from boltons.iterutils import chunked_iter

from langml.baselines import BaseDataLoader
from langml.baselines.contrastive.utils import aeda_augment, whitespace_tokenize
from langml import TF_KERAS
if TF_KERAS:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
else:
    from keras.preprocessing.sequence import pad_sequences


class DataLoader(BaseDataLoader):
    def __init__(self,
                 data: List,
                 tokenizer: object,
                 batch_size: int = 32):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    @staticmethod
    def load_data(fpath: str,
                  apply_aeda: bool = True,
                  aeda_tokenize: Callable = whitespace_tokenize,
                  aeda_language: str = 'EN') -> Tuple[
                      List[Tuple[str, str]], List[Tuple[str, str, int]]]:
        """
        Args:
          fpath: str, path of data
          apply_aeda: bool, whether to apply the AEDA technique to augment data, default True
          aeda_tokenize: Callable, specify aeda tokenize function, it works when set apply_aeda=True
          aeda_language: str, specifying the language, it works when set apply_aeda=True
        """
        data, data_with_label = [], []
        with open(fpath, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if 'text_left' in obj:
                    data_with_label.append((obj['text_left'], obj['text_right'], int(obj['label'])))
                    texts = [obj['text_left'], obj['text_right']]
                else:
                    texts = [obj['text']]
                for text in texts:
                    if apply_aeda:
                        augmented_text = aeda_augment(aeda_tokenize(text), language=aeda_language)
                    else:
                        augmented_text = text
                    data.append((text, augmented_text))
        return data, data_with_label

    def make_iter(self, random: bool = False):
        if random:
            shuffle(self.data)

        for chunks in chunked_iter(self.data, self.batch_size):
            batch_tokens, batch_segments, batch_augmented_tokens, batch_augmented_segments = [], [], [], []
            for text, augmented_text in chunks:
                tokenized = self.tokenizer.encode(text)
                batch_tokens.append(tokenized.ids)
                batch_segments.append(tokenized.segment_ids)
                tokenized = self.tokenizer.encode(augmented_text)
                batch_augmented_tokens.append(tokenized.ids)
                batch_augmented_segments.append(tokenized.segment_ids)

            batch_tokens = pad_sequences(batch_tokens, padding='post', truncating='post')
            batch_segments = pad_sequences(batch_segments, padding='post', truncating='post')
            batch_augmented_tokens = pad_sequences(batch_augmented_tokens, padding='post', truncating='post')
            batch_augmented_segments = pad_sequences(batch_augmented_segments, padding='post', truncating='post')
            batch_labels = np.zeros((len(batch_tokens), 1))

            yield [batch_tokens, batch_segments, batch_augmented_tokens, batch_augmented_segments], batch_labels


class TFDataLoader(DataLoader):
    def make_iter(self, random: bool = False):
        def gen_features():
            for text, augmented_text in self.data:
                tokenized = self.tokenizer.encode(text)
                d = {
                    'Input-Token': tokenized.ids,
                    'Input-Segment': tokenized.segment_ids
                }
                tokenized = self.tokenizer.encode(augmented_text)
                d = dict(d, **{
                    'Input-Augmented-Token': tokenized.ids,
                    'Input-Augmented-Segment': tokenized.segment_ids
                })
                yield d, [0]

        d = {
            'Input-Token': tf.int64,
            'Input-Segment': tf.int64,
            'Input-Augmented-Token': tf.int64,
            'Input-Augmented-Segment': tf.int64
        }
        output_types = (d, tf.int64)
        d = {
            'Input-Token': tf.TensorShape((None, )),
            'Input-Segment': tf.TensorShape((None, )),
            'Input-Augmented-Token': tf.TensorShape((None, )),
            'Input-Augmented-Segment': tf.TensorShape((None, ))
        }
        output_shapes = (d, tf.TensorShape((1, )))
        dataset = tf.data.Dataset.from_generator(gen_features,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes)
        dataset = dataset.repeat()
        if random:
            dataset = dataset.shuffle(self.batch_size * 1000)
        dataset = dataset.padded_batch(self.batch_size, output_shapes).prefetch(self.batch_size * 1000)
        return dataset

    def __call__(self, random: bool = False):
        return self.make_iter(random=random)
