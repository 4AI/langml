# -*- coding: utf-8 -*-

import random
from typing import List


CN_PUNCTUATIONS = ['。', '，', '？', '！', '；']
EN_PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']


def aeda_augment(words: List[str], ratio: float = 0.3, language: str = 'EN') -> str:
    """ AEDA：An Easier Data Augmentation Technique for Text Classification
    Args:
      text: str, input text
      ratio: float, ratio to add punctuation randomly
      language: str, specify language from ['EN', 'CN'], default EN
    """
    assert language in ['EN', 'CN'], 'please specify language from ["EN", "CN"]'
    punctuations = EN_PUNCTUATIONS if language == 'EN' else CN_PUNCTUATIONS
    join_str = ' ' if language == 'EN' else ''

    q = random.randint(1, int(ratio * len(words) + 1))
    qs = random.sample(range(len(words)), q)
    augmented_words = []
    for i, word in enumerate(words):
        if i in qs:
            augmented_words.append(random.choice(punctuations))
        augmented_words.append(word)
    return join_str.join(augmented_words)


def whitespace_tokenize(text: str) -> List[str]:
    return text.split()
