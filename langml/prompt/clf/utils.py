# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import numpy as np
from boltons.iterutils import chunked_iter
from sklearn.metrics import classification_report, f1_score

from langml import keras, TF_KERAS
from langml.prompt.base import Template


def merge_template_tokens(
    template_ids: List[int],
    token_ids: List[int],
    max_length: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """ Merge template and token ids
    Args:
      - template_ids: List[int], template ids
      - token_ids: List[int], token ids
      - max_length: int, max length
    Return:
      - token_ids: List[int], merged token ids
      - template_mask: List[int], template mask
    """
    token_ids = [token_ids[0]] + template_ids + token_ids[1:-1]
    if max_length:
        token_ids = token_ids[:max_length - 1] + [token_ids[-1]]
    template_mask = [0] + [1] * len(template_ids) + [0] * (len(token_ids) - len(template_ids) - 1)
    return token_ids,  template_mask


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self,
                 data: List[str],
                 labels: List[str],
                 mask_id: int,
                 template: Template,
                 patience: int = 10,
                 batch_size: int = 32,
                 model_path: Optional[str] = None,
                 f1_average: str = 'macro'):
        self.data = data
        self.labels = labels
        self.mask_id = mask_id
        self.template = template
        self.patience = patience
        self.batch_size = batch_size
        self.model_path = model_path
        self.f1_average = f1_average

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best_f1 = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        pred_labels = []
        for chunk in chunked_iter(self.data, self.batch_size):
            batch_templates, batch_tokens, batch_segments, batch_mask_ids = [], [], [], []
            for obj in chunk:
                batch_templates.append(obj['template_mask'])
                batch_tokens.append(obj['token_ids'])
                batch_segments.append(obj['segment_ids'])
                batch_mask_ids.append(obj['mask_ids'])

            batch_templates = keras.preprocessing.sequence.pad_sequences(
                batch_templates, truncating='post', padding='post')
            batch_tokens = keras.preprocessing.sequence.pad_sequences(
                batch_tokens, truncating='post', padding='post')
            batch_segments = keras.preprocessing.sequence.pad_sequences(
                batch_segments, truncating='post', padding='post')
            batch_mask_ids = keras.preprocessing.sequence.pad_sequences(
                batch_mask_ids, truncating='post', padding='post')

            if TF_KERAS:
                logits = self.model([batch_templates, batch_tokens, batch_segments, batch_mask_ids])
            else:
                logits = self.model.predict([batch_templates, batch_tokens, batch_segments, batch_mask_ids])

            output = np.argmax(logits[0], axis=1)
            output = output * (batch_tokens == self.mask_id).astype('int')
            output = output[output > 0].tolist()

            pred_labels += [self.template.decode_label(idx) for idx in output]

        assert len(self.labels) == len(pred_labels)
        print(classification_report(self.labels, pred_labels))
        f1 = f1_score(self.labels, pred_labels, average=self.f1_average)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
            if self.model_path is not None:
                print(f'new best model, save weights to {self.model_path}')
                self.model.save_weights(self.model_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
