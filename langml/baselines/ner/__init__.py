# -*- coding: utf-8 -*-

import re
from typing import Dict, List, Optional

import numpy as np
from seqeval.metrics import classification_report

from langml import TF_VERSION
from langml.utils import bio_decode
from langml.tensor_typing import Models


re_split = re.compile(r'.*?[\n。]+')


class Infer:
    def __init__(self,
                 model: Models,
                 tokenizer: object,
                 id2label: Dict,
                 max_chunk_len: Optional[int] = None,
                 is_bert: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_chunk_len = max_chunk_len
        self.is_bert = is_bert

    def decode_one(self, text: str, base_position: int = 0):
        """
        Args:
          - text: str
          - base_position: int

        Return:
          list of tuple: [(entity, start, end, entity_type)]
        """
        tokened = self.tokenizer.encode(text)
        mapping = self.tokenizer.tokens_mapping(text, tokened.tokens)
        token_ids = np.array([tokened.ids])
        segment_ids = np.array([tokened.segment_ids])
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
        tags = [self.id2label[i] for i in np.argmax(logits[0], axis=1)]
        entities = bio_decode(tags)

        res = []
        for s, e, t in entities:
            s = mapping[s]
            e = mapping[e]
            s = 0 if not s else s[0]
            e = len(text) - 1 if not e else e[-1]
            res.append((base_position + s, base_position + e + 1, text[s: e + 1], t))
        return res

    def __call__(self, text: str):
        if self.max_chunk_len is None or len(text) < self.max_chunk_len:
            return self.decode_one(text)
        sentences = re_split.findall(text)
        if not sentences:
            return self.decode_one(text)
        results = []
        prev, base_position = 0, 0
        for i in range(1, len(sentences)):
            current_text = ''.join(sentences[prev: i])
            if len(current_text) <= self.max_chunk_len and len(''.join(sentences[prev: i+1])) > self.max_chunk_len:
                results.extend(self.decode_one(current_text, base_position=base_position))
                prev = i
                base_position += len(current_text)
        results.extend(self.decode_one(''.join(sentences[prev:]), base_position=base_position))
        return results


def compute_detail_metrics(model: Models, dataloader: object, id2label: Dict, is_bert: bool = True):
    all_preds, all_golds = [], []
    for pairs in dataloader.data:
        token_ids, segment_ids, labels = dataloader.encode_data(pairs)
        token_ids = np.array([token_ids])
        segment_ids = np.array([segment_ids])
        if TF_VERSION > 1:
            if is_bert:
                logits = model([token_ids, segment_ids])
            else:
                logits = model([token_ids])
        else:
            if is_bert:
                logits = model.predict([token_ids, segment_ids])
            else:
                logits = model.predict([token_ids])
        gold_tags = [id2label[i] for i in labels]
        pred_tags = [id2label[i] for i in np.argmax(logits[0], axis=1)]
        assert len(gold_tags) == len(pred_tags), f'g: {len(gold_tags)}, t: {len(pred_tags)}'
        all_preds.append(pred_tags)
        all_golds.append(gold_tags)
    
    print(classification_report(all_golds, all_preds, digits=4))
