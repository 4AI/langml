# -*- coding: utf-8 -*-

from typing import List, Optional
from random import choice

import numpy as np

from langml import keras, TF_KERAS
from langml.prompt.base import BasePromptTask, BaseDataGenerator, Template
from langml.tokenizer import Tokenizer
from langml.log import info
from langml.prompt.clf.utils import MetricsCallback, merge_template_tokens


class DataGenerator(BaseDataGenerator):
    def __init__(self,
                 data: List[str],
                 labels: List[str],
                 tokenizer: Tokenizer,
                 template: Template,
                 batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.mask_id = self.tokenizer.token_to_id(self.tokenizer.special_tokens.MASK)

        self.data = []
        for data, label in zip(data, labels):
            tokened = tokenizer.encode(data)
            token_ids, template_mask = merge_template_tokens(template.template_ids, tokened.ids, tokenizer.max_length)
            segment_ids = [0] * len(token_ids)
            mask_ids = (np.array(token_ids) == self.mask_id).astype('int')
            assert mask_ids.sum() == 1
            output_ids = [choice(template.label2tokens[label]) if t == self.mask_id else t for t in token_ids]
            self.data.append({
                'template_mask': template_mask,
                'token_ids': token_ids,
                'segment_ids': segment_ids,
                'mask_ids': mask_ids,
                'output_ids': output_ids,
            })

        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def make_iter(self, random: bool = False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_templates, batch_tokens, batch_segments, batch_mask_ids, batch_outputs = [], [], [], [], []
        for idx in idxs:
            obj = self.data[idx]
            batch_templates.append(obj['template_mask'])
            batch_tokens.append(obj['token_ids'])
            batch_segments.append(obj['segment_ids'])
            batch_mask_ids.append(obj['mask_ids'])
            batch_outputs.append(obj['output_ids'])

            if len(batch_tokens) == self.batch_size or idx == idxs[-1]:
                batch_templates = keras.preprocessing.sequence.pad_sequences(
                    batch_templates, truncating='post', padding='post')
                batch_outputs = keras.preprocessing.sequence.pad_sequences(
                    batch_outputs, truncating='post', padding='post')
                batch_outputs = np.expand_dims(batch_outputs, axis=-1)
                batch_tokens = keras.preprocessing.sequence.pad_sequences(
                    batch_tokens, truncating='post', padding='post')
                batch_segments = keras.preprocessing.sequence.pad_sequences(
                    batch_segments, truncating='post', padding='post')
                batch_mask_ids = keras.preprocessing.sequence.pad_sequences(
                    batch_mask_ids, truncating='post', padding='post')
                yield [batch_templates, batch_tokens, batch_segments, batch_mask_ids], [batch_outputs]
                batch_templates, batch_tokens, batch_segments, batch_mask_ids, batch_outputs = [], [], [], [], []


class PTuningForClassification(BasePromptTask):
    def fit(self,
            data: List[str],
            labels: List[str],
            valid_data: Optional[List[str]] = None,
            valid_labels: Optional[List[str]] = None,
            model_path: Optional[str] = None,
            epoch: int = 20,
            batch_size: int = 16,
            early_stop: int = 10,
            do_shuffle: bool = True,
            f1_average: str = 'macro',
            verbose: int = 1):
        """ Fitting ptuning model for classification
        Args:
          - data: List[str], texts of traning data
          - labels: List[Union[str, List[str]]], traning labels
          - valid_data: List[str], texts of valid data
          - valid_labels: List[Union[str, List[str]]], labels of valid data
          - model_path: Optional[str], path to save model, default `None`, do not to save model
          - epoch: int, epochs to train
          - batch_size: int, batch size,
          - early_stop: int, patience of early stop
          - do_shuffle: whether to shuffle data in training phase
          - f1_average: str, {'micro', 'macro', 'samples','weighted', 'binary'} or None
          - verbose: int, 0 = silent, 1 = progress bar, 2 = one line per epoch
        """
        assert len(data) == len(labels), "data size should equal to label size"

        generator = DataGenerator(data, labels, self.tokenizer, self.template, batch_size=batch_size)
        callbacks = []
        if valid_data is not None and valid_labels is not None:
            valid_generator = DataGenerator(valid_data, valid_labels, self.tokenizer, self.template)
            callback = MetricsCallback(valid_generator.data, valid_labels,
                                       self.mask_id, self.template,
                                       patience=early_stop, batch_size=batch_size,
                                       model_path=model_path, f1_average=f1_average)
            callbacks.append(callback)

        info('start to train...')
        self.model.fit(
            generator(random=do_shuffle),
            steps_per_epoch=len(generator),
            epochs=epoch,
            verbose=verbose,
            callbacks=callbacks if callbacks else None,
        )
        if not callbacks:
            self.model.save_weights(model_path)

    def predict(self, text: str) -> str:
        tokenized = self.tokenizer.encode(text)
        token_ids, template_mask = merge_template_tokens(
            self.template.template_ids, tokenized.ids, self.tokenizer.max_length)
        token_ids = np.array([token_ids])
        segment_ids = np.zeros_like(token_ids)
        template_mask = np.array([template_mask])
        mask_ids = (token_ids == self.mask_id).astype('int')

        if TF_KERAS:
            logits = self.model([template_mask, token_ids, segment_ids, mask_ids])
        else:
            logits = self.model.predict([template_mask, token_ids, segment_ids, mask_ids])

        output = np.argmax(logits[0], axis=1)
        output = output * (token_ids == self.mask_id).astype('int')
        output = output[output > 0].tolist()

        return self.template.decode_label(output[0])

    def load(self, model_path: str):
        """ load model
        Args:
          - model_path: str, model path
        """
        self.model.load_weights(model_path)
