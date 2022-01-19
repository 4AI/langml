# -*- coding: utf-8 -*-

import json
import math
from random import shuffle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from boltons.iterutils import chunked_iter

from langml.baselines import BaseDataLoader
from langml.utils import pad_sequences


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
                  build_vocab: bool = False,
                  label2idx: Optional[Dict] = None) -> Union[
                      List[Tuple[str, str, int]], Tuple[List[Tuple[str, str, int]], Dict]]:
        """
        Args:
          fpath: str, path of data
          build_vocab: bool, whether to build vocabulary
          label2idx: Optional[Dict], label to index dict
        """
        if build_vocab:
            label_set = set()

        raw_data = []
        with open(fpath, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if build_vocab:
                    label_set.add(obj['label'])
                raw_data.append((obj['text_left'], obj['text_right'], obj['label']))

        if build_vocab:
            labels = list(label_set)
            # to compute Spearman's Rank Correlation Coefficient, labels must be sorted.
            labels.sort()
            label2idx = dict(zip(labels, range(len(labels))))

        data = []
        for text_left, text_right, label in raw_data:
            if label2idx is not None:
                label = label2idx.get(label, int(label))
            else:
                label = float(label)
            data.append((text_left, text_right, label))

        if build_vocab:
            return data, label2idx
        return data

    def make_iter(self, random: bool = False):
        if random:
            shuffle(self.data)

        for chunks in chunked_iter(self.data, self.batch_size):
            batch_left_tokens, batch_left_segments = [], []
            batch_right_tokens, batch_right_segments = [], []
            batch_labels = []
            for text_left, text_right, label in chunks:
                tokenized = self.tokenizer.encode(text_left)
                batch_left_tokens.append(tokenized.ids)
                batch_left_segments.append(tokenized.segment_ids)
                tokenized = self.tokenizer.encode(text_right)
                batch_right_tokens.append(tokenized.ids)
                batch_right_segments.append(tokenized.segment_ids)
                batch_labels.append([label])

            batch_left_tokens = pad_sequences(batch_left_tokens, padding='post', truncating='post')
            batch_left_segments = pad_sequences(batch_left_segments, padding='post', truncating='post')
            batch_right_tokens = pad_sequences(batch_right_tokens, padding='post', truncating='post')
            batch_right_segments = pad_sequences(batch_right_segments, padding='post', truncating='post')
            batch_labels = np.array(batch_labels)

            yield [batch_left_tokens, batch_left_segments, batch_right_tokens, batch_right_segments], batch_labels


class TFDataLoader(DataLoader):
    def make_iter(self, random: bool = False):
        def gen_features():
            for text_left, text_right, label in self.data:
                tokenized = self.tokenizer.encode(text_left)
                d = {
                    'Input-Token': tokenized.ids,
                    'Input-Segment': tokenized.segment_ids
                }
                tokenized = self.tokenizer.encode(text_right)
                d = dict(d, **{
                    'Input-Right-Token': tokenized.ids,
                    'Input-Right-Segment': tokenized.segment_ids
                })
                yield d, [label]

        d = {
            'Input-Token': tf.int64,
            'Input-Segment': tf.int64,
            'Input-Right-Token': tf.int64,
            'Input-Right-Segment': tf.int64,
        }
        output_types = (d, tf.int64)
        d = {
            'Input-Token': tf.TensorShape((None, )),
            'Input-Segment': tf.TensorShape((None, )),
            'Input-Right-Token': tf.TensorShape((None, )),
            'Input-Right-Segment': tf.TensorShape((None, )),
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
