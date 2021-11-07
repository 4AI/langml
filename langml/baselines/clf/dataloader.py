# -*- coding: utf-8 -*-

import json
import math
from random import shuffle
from typing import Dict, List

import numpy as np
from boltons.iterutils import chunked_iter
import tensorflow as tf
from langml.baselines import BaseDataLoader
from langml import TF_KERAS
if TF_KERAS:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
else:
    from keras.preprocessing.sequence import pad_sequences


class DataLoader(BaseDataLoader):
    def __init__(self,
                 data: List,
                 tokenizer: object,
                 label2id: Dict,
                 batch_size: int = 32,
                 is_bert: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.is_bert = is_bert
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    @staticmethod
    def load_data(fpath: str, build_vocab: bool = False) -> List:
        if build_vocab:
            label2id = {}
        data = []
        with open(fpath, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if build_vocab and obj['label'] not in label2id:
                    label2id[obj['label']] = len(label2id)
                data.append((obj['text'], obj['label']))
        if build_vocab:
            return data, label2id
        return data

    def make_iter(self, random: bool = False):
        if random:
            shuffle(self.data)

        for chunks in chunked_iter(self.data, self.batch_size):
            batch_tokens, batch_segments, batch_labels = [], [], []
            for text, label in chunks:
                tokenized = self.tokenizer.encode(text)
                batch_tokens.append(tokenized.ids)
                batch_segments.append(tokenized.segment_ids)
                batch_labels.append([self.label2id[label]])

            batch_tokens = pad_sequences(batch_tokens, padding='post', truncating='post')
            batch_segments = pad_sequences(batch_segments, padding='post', truncating='post')
            batch_labels = np.array(batch_labels)
            if self.is_bert:
                yield [batch_tokens, batch_segments], batch_labels
            else:
                yield batch_tokens, batch_labels


class TFDataLoader(DataLoader):
    def make_iter(self, random: bool = False):
        def gen_features():
            for text, label in self.data:
                tokenized = self.tokenizer.encode(text)
                if self.is_bert:
                    yield {'Input-Token': tokenized.ids,
                           'Input-Segment': tokenized.segment_ids}, [self.label2id[label]]
                else:
                    yield tokenized.ids, [label]

        if self.is_bert:
            output_types = ({'Input-Token': tf.int64, 'Input-Segment': tf.int64}, tf.int64)
            output_shapes = ({'Input-Token': tf.TensorShape((None, )),
                              'Input-Segment': tf.TensorShape((None, ))},
                             tf.TensorShape((1, )))
        else:
            output_types = (tf.int64, tf.int64)
            output_shapes = (tf.TensorShape((None, )), tf.TensorShape((1, )))
        dataset = tf.data.Dataset.from_generator(gen_features,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes)
        dataset = dataset.repeat()
        if random:
            dataset = dataset.shuffle(self.batch_size * 1000)
        dataset = dataset.padded_batch(self.batch_size, output_shapes)
        return dataset

    def __call__(self, random: bool = False):
        return self.make_iter(random=random)
