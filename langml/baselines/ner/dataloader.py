# -*- coding: utf-8 -*-

import math
from random import shuffle
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from boltons.iterutils import chunked_iter
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
                 max_len: Optional[int] = None,
                 is_bert: bool = True):
        self.data = data
        self.label2id = label2id
        self.batch_size = batch_size
        self.max_len = max_len
        self.is_bert = is_bert
        self.tokenizer = tokenizer
        self.start_token_id = tokenizer.token_to_id(tokenizer.special_tokens.CLS)
        self.end_token_id = tokenizer.token_to_id(tokenizer.special_tokens.SEP)

    def encode_data(self, data: List[Tuple[str, str]]) -> Tuple[List[int], List[int], List[int]]:
        token_ids, labels = [self.start_token_id], [self.label2id['O']]
        for segment, label in data:
            tokenized = self.tokenizer.encode(segment)
            token_id = tokenized.ids[1:-1]
            token_ids += token_id
            if label == 'O':
                labels += [self.label2id['O']] * len(token_id)
            else:
                labels += ([self.label2id[f'B-{label}']] + [self.label2id[f'I-{label}']] * (len(token_id) - 1))
        assert len(token_ids) == len(labels)
        if self.max_len is not None:
            token_ids = token_ids[:self.max_len - 1]
            labels = labels[:self.max_len - 1]
        token_ids += [self.end_token_id]
        labels += [self.label2id['O']]
        segment_ids = [0] * len(token_ids)
        return token_ids, segment_ids, labels

    @staticmethod
    def load_data(fpath: str, build_vocab: bool = False) -> List:
        if build_vocab:
            label2id = {'O': 0}
        data = []
        with open(fpath, 'r', encoding='utf-8') as reader:
            for sentence in reader.read().split('\n\n'):
                if not sentence:
                    continue
                current_data = []
                for chunk in sentence.split('\n'):
                    try:
                        segment, label = chunk.split('\t')
                        if build_vocab:
                            if label != 'O' and f'B-{label}' not in label2id:
                                label2id[f'B-{label}'] = len(label2id)
                            if label != 'O' and f'I-{label}' not in label2id:
                                label2id[f'I-{label}'] = len(label2id)
                        current_data.append((segment, label))
                    except ValueError:
                        print('broken data:', chunk)
                data.append(current_data)
        if build_vocab:
            return data, label2id
        return data

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def make_iter(self, random: bool = False):
        if random:
            shuffle(self.data)

        for chunks in chunked_iter(self.data, self.batch_size):
            batch_tokens, batch_segments, batch_labels = [], [], []

            for chunk in chunks:
                token_ids, segment_ids, labels = self.encode_data(chunk)
                batch_tokens.append(token_ids)
                batch_segments.append(segment_ids)
                batch_labels.append(labels)

            batch_tokens = pad_sequences(batch_tokens, padding='post', truncating='post')
            batch_segments = pad_sequences(batch_segments, padding='post', truncating='post')
            batch_labels = pad_sequences(batch_labels, padding='post', truncating='post')
            if self.is_bert:
                yield [batch_tokens, batch_segments], batch_labels
            else:
                yield batch_tokens, batch_labels


class TFDataLoader(DataLoader):
    def make_iter(self, random: bool = False):
        def gen_features():
            for pairs in self.data:
                token_ids, segment_ids, labels = self.encode_data(pairs)
                if self.is_bert:
                    yield {'Input-Token': token_ids,
                           'Input-Segment': segment_ids}, labels
                else:
                    yield token_ids, labels

        if self.is_bert:
            output_types = ({'Input-Token': tf.int64, 'Input-Segment': tf.int64}, tf.int64)
            output_shapes = ({'Input-Token': tf.TensorShape((None, )),
                              'Input-Segment': tf.TensorShape((None, ))},
                             tf.TensorShape((None, )))
        else:
            output_types = (tf.int64, tf.int64)
            output_shapes = (tf.TensorShape((None, )), tf.TensorShape((None, )))
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
