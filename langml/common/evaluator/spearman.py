# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

from langml.utils import pad_sequences
from langml.tensor_typing import Models
from langml.tokenizer import Tokenizer


class SpearmanEvaluator:
    def __init__(self, encoder: Models, tokenizer: Tokenizer) -> None:
        self.encoder = encoder
        self.tokenizer = tokenizer

    def compute_corrcoef(self, data: List[Tuple[str, str, int]]) -> float:
        left_token_ids = []
        right_token_ids = []
        labels = []
        for text_left, text_right, label in data:
            tokenized = self.tokenizer.encode(text_left)
            left_token_ids.append(tokenized.ids)

            tokenized = self.tokenizer.encode(text_right)
            right_token_ids.append(tokenized.ids)
            labels.append(float(label))

        left_token_ids = pad_sequences(left_token_ids, padding='post', truncating='post')
        right_token_ids = pad_sequences(right_token_ids, padding='post', truncating='post')
        left_vecs = self.encoder.predict([left_token_ids, np.zeros_like(left_token_ids)], verbose=True)
        right_vecs = self.encoder.predict([right_token_ids, np.zeros_like(right_token_ids)], verbose=True)
        left_vecs = normalize(left_vecs, norm='l2')
        right_vecs = normalize(right_vecs, norm='l2')
        similarity = (left_vecs * right_vecs).sum(axis=1)

        return spearmanr(labels, similarity).correlation
