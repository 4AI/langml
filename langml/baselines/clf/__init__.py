# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Tuple, Union
    
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

from langml import TF_VERSION
from langml.tensor_typing import Models


class Infer:
    def __init__(self,
                 model: Models,
                 tokenizer: object,
                 id2label: Dict,
                 is_bert: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.is_bert = is_bert

    def __call__(self, text: str):
        tokenized = self.tokenizer.encode(text)
        token_ids = np.array([tokenized.ids])
        segment_ids = np.array([tokenized.segment_ids])
        if TF_VERSION > 1:
            if self.is_bert:
                logits = self.model([token_ids, segment_ids])
            else:
                logits = self.model([token_ids])
        else:
            if self.is_bert:
                logits = self.model.predict([token_ids, segment_ids])
            else:
                logits = self.model.predict([token_ids])
        pred = np.argmax(logits, 1)[0]
        return self.id2label[pred]


def compute_detail_metrics(infer: object, datas: List, use_micro=False) -> Tuple[float, float, Union[str, Dict]]:
    y_true, y_pred = [], []
    for text, label in datas:
        y_pred.append(infer(text))
        y_true.append(label)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro' if use_micro else 'macro')
    cr = classification_report(y_true=y_true, y_pred=y_pred)
    return f1, acc, cr
