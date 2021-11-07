# -*- coding: utf-8 -*-

"""
LangML Tokenizer

- WPTokenizer: WordPiece Tokenizer
- SPTokenizer: SentencePiece Tokenizer

Wrap for:
    - tokenizers.BertWordPieceTokenizer
    - sentencepiece.SentencePieceProcessor

We don't provide all functions of raw tokenizer, please use raw tokenizer for full usage.
"""

import unicodedata
from math import ceil
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sentencepiece import SentencePieceProcessor
from tokenizers import BertWordPieceTokenizer


class Encoding:
    ''' Product of tokenizer encoding
    '''
    ids = None
    segment_ids = None
    tokens = None

    def __init__(self,
                 ids: Union[np.ndarray, List[int]],
                 segment_ids: Union[np.ndarray, List[int]],
                 tokens: List[str]) -> None:
        self.ids = ids
        self.segment_ids = segment_ids
        self.tokens = tokens


class SpecialTokens:
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'
    CLS = '[CLS]'
    SEP = '[SEP]'

    def __contains__(self, token: str) -> bool:
        """ Check if the input token exists in special tokens.
        Args:
          - token: str
        Return:
          bool
        """
        return token in [
            self.PAD, self.UNK, self.MASK, self.CLS, self.SEP
        ]

    def tokens(self) -> List[str]:
        ret = []
        for field in SpecialTokens.__dict__.keys():
            if field.startswith('_'):
                continue
            if isinstance(getattr(self, field), str):
                ret.append(getattr(self, field))
        return ret


class Tokenizer(metaclass=ABCMeta):
    """ Base Tokenizer
    """

    def __init__(self, vocab_path: str, lowercase: bool = False):
        """
        Args:
          - vocab_path: str, path to vocab
          - lowercase: bool, whether to do lowercase
        """
        self.vocab_path = vocab_path
        self.lowercase = lowercase
        self.special_tokens = SpecialTokens()

        self.max_length = None
        self.truncation_strategy = None
        self._tokenizer = None

    def enable_truncation(self, max_length: int, strategy: str = 'post'):
        """
        Args:
          - max_length: int,
          - strategy: str, optional, truncation strategy, options: `post` or `pre`, default `post`
        """
        self.max_length = max_length
        self.truncation_strategy = strategy
        if strategy is not None:
            assert self.truncation_strategy in ['post', 'pre'], '`strategy` must be `post` or `pre`'

    def tokens_mapping(self, sequence: str, tokens: List[str]) -> List[Tuple[int, int]]:
        """ Get tokens to their corresponding sequence position mapping.
        Tokens may contain special marks, e.g., `##`, `▁`, and `[UNK]`.
        Use this function can obtain the corresponding raw token in the sequence.

        Args:
          - sequence: str, the input sequence
          - tokens: List[str], tokens of the input sequence
        Return:
          List[Tuple[int, int]]

        Examples:
        >>> sequence = 'I like watermelons'
        >>> tokens = ['[CLS]', '▁i', '▁like', '▁water', 'mel', 'ons', '[SEP]']
        >>> mapping = tokenizer.tokens_mapping(tokens)
        >>> start_index, end_index = 3, 5
        >>> print("current token", tokens[start_index: end_index + 1])
        ['▁water', 'mel', 'ons']
        >>> print("raw token", sequence[mapping[start_index][0]: mapping[end_index][1]])
        watermelons

        Reference:
          https://github.com/bojone/bert4keras
        """
        if self.lowercase:
            sequence = self.sequence_lower(sequence)

        normalized_sequence, char_mapping = '', []
        for i, ch in enumerate(sequence):
            if self.lowercase:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or (unicodedata.category(ch) in ('Cc', 'Cf')))
            ])
            normalized_sequence += ch
            char_mapping.extend([i] * len(ch))

        sequence = normalized_sequence
        mapping = []
        offset = 0
        special_placeholder = (0, 0)
        for token in tokens:
            if token in self.special_tokens:
                mapping.append(special_placeholder)
            else:
                token = self.stem(token)
                start = sequence[offset:].index(token) + offset
                end = start + len(token)
                cnt = char_mapping[start:end]
                mapping.append((cnt[0], cnt[-1] + 1))
                offset = end

        return mapping

    def encode(self, sequence: str, pair: Optional[str] = None, return_array: bool = False) -> Encoding:
        """
        Args:
          - sequence: str, input sequence
          - pair: str, optional, pair sequence, default `None`
          - return_array: bool, optional, whether to return numpy array, default `True`
        Return:
          Encoding object
        """
        if self.lowercase:
            sequence = self.sequence_lower(sequence)
            if pair:
                pair = self.sequence_lower(pair)
        tokens = self.tokenize(sequence)
        pair_tokens = None
        if pair is not None:
            pair_tokens = self.tokenize(pair)

        if self.max_length is not None:
            max_token_length = self.max_length - 2
            if pair_tokens is not None:
                max_token_length -= 1
            tokens, pair_tokens = self.sequence_truncating(max_token_length, tokens, pair_tokens)

        tokens = [self.special_tokens.CLS] + tokens + [self.special_tokens.SEP]
        token_ids = [self.token_to_id(token) for token in tokens]
        segment_ids = [0] * len(token_ids)

        if pair_tokens is not None:
            pair_tokens = pair_tokens + [self.special_tokens.SEP]
            pair_token_ids = [self.token_to_id(token) for token in pair_tokens]
            pair_segment_ids = [1] * len(pair_token_ids)

            tokens += pair_tokens
            token_ids += pair_token_ids
            segment_ids += pair_segment_ids

        if return_array:
            token_ids = np.array(token_ids)
            segment_ids = np.array(segment_ids)

        return Encoding(
            ids=token_ids,
            segment_ids=segment_ids,
            tokens=tokens,
        )

    def encode_batch(self,
                     inputs: Union[List[str], List[Tuple[str, str]], List[List[str]]],
                     padding: bool = True,
                     padding_strategy: str = 'post',
                     return_array: bool = False) -> Encoding:
        """
        Args:
          - inputs: Union[List[str], List[Tuple[str, str]], List[List[str]]], list of texts or list of text pairs.
          - padding: bool, optional, whether to padding sequences, default `True`
          - padding_strategy: str, optional, options: `post` or `pre`, default `post`
          - return_array: bool, optional, whether to return numpy array, default `True`
        Return:
          Encoding object
        """
        assert padding_strategy in ['post', 'pre'], '`padding_strategy` must be `post` or `pre`'
        all_tokens, all_pair_tokens = [], []
        for item in inputs:
            if isinstance(item, (tuple, list)):
                assert len(item) == 2
                item = list(item)
                if self.lowercase:
                    item[0] = self.sequence_lower(item[0])
                    item[1] = self.sequence_lower(item[1])
                all_tokens.append(self.tokenize(item[0]))
                all_pair_tokens.append(self.tokenize(item[1]))
            elif isinstance(item, str):
                if self.lowercase:
                    item = self.sequence_lower(item)
                all_tokens.append(self.tokenize(item))

        if not all_pair_tokens:
            all_pair_tokens = None

        max_all_token_length = max(len(t) for t in all_tokens)
        if all_pair_tokens is not None:
            max_all_token_pair_length = max(len(t) + len(p) for t, p in zip(all_tokens, all_pair_tokens))

        if self.max_length is not None:
            max_token_length = self.max_length - 2
            if all_pair_tokens is not None:
                max_token_length -= 1
                max_token_length = min(max_token_length, max_all_token_pair_length)
            else:
                max_token_length = min(max_token_length, max_all_token_length)
        else:
            if all_pair_tokens is not None:
                max_token_length = max_all_token_pair_length
            else:
                max_token_length = max_all_token_length

        batch_tokens = []
        batch_token_ids = []
        batch_segment_ids = []
        all_pair_tokens = all_pair_tokens or [None] * len(all_tokens)
        for tokens, pair_tokens in zip(all_tokens, all_pair_tokens):
            tokens, pair_tokens = self.sequence_truncating(max_token_length, tokens, pair_tokens)
            repeat = 0
            if padding:
                if pair_tokens is not None:
                    repeat = max_token_length - len(tokens) - len(pair_tokens)
                else:
                    repeat = max_token_length - len(tokens)

            tokens = [self.special_tokens.CLS] + tokens + [self.special_tokens.SEP]
            token_ids = [self.token_to_id(token) for token in tokens]
            segment_ids = [0] * len(token_ids)

            if pair_tokens is not None:
                pair_tokens = pair_tokens + [self.special_tokens.SEP]
                pair_token_ids = [self.token_to_id(token) for token in pair_tokens]
                pair_segment_ids = [1] * len(pair_token_ids)

                tokens += pair_tokens
                token_ids += pair_token_ids
                segment_ids += pair_segment_ids

            if padding and repeat > 0:
                padding_value = self.token_to_id(self.special_tokens.PAD)
                if padding_strategy == 'post':
                    tokens += [self.special_tokens.PAD] * repeat
                    token_ids += [padding_value] * repeat
                    segment_ids += [padding_value] * repeat
                elif padding_strategy == 'pre':
                    tokens = [self.special_tokens.PAD] * repeat + tokens
                    token_ids = [padding_value] * repeat + token_ids
                    segment_ids = [padding_value] * repeat + segment_ids

            batch_tokens.append(tokens)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

        if return_array:
            batch_token_ids = np.array(batch_token_ids)
            batch_segment_ids = np.array(batch_segment_ids)

        return Encoding(
            ids=batch_token_ids,
            segment_ids=batch_segment_ids,
            tokens=batch_tokens
        )

    def stem(self, token):
        if isinstance(self, WPTokenizer) and token.startswith('##'):
            return token[2:]
        elif isinstance(self, SPTokenizer) and token.startswith('▁'):
            return token[1:]
        return token

    def sequence_lower(self, sequence: str) -> str:
        """ Do lower to sequence, except for special tokens.
        Args:
          - sequence: str
        Return:
          str
        """
        sequence = sequence.lower()
        for token in self.special_tokens.tokens():
            sequence = sequence.replace(token.lower(), token)
        return sequence

    def sequence_truncating(self,
                            max_token_length: int,
                            tokens: List[str],
                            pair_tokens: Optional[List[str]] = None) -> Tuple[
                                List[str], Optional[List[str]]]:
        """ Truncating sequence
        Args:
          - max_token_length: int, maximum token length
          - tokens: List[str], input tokens
          - pair_tokens:  Optional[List[str]], optional, input pair tokens, default None
        Return:
          Tuple[List[str], Optional[List[str]]]
        """
        if pair_tokens is not None:
            left_len = len(tokens)
            right_len = len(pair_tokens)
            if left_len + right_len <= max_token_length:
                max_left = left_len
                max_right = right_len
            else:
                max_left = min(ceil(max_token_length / 2), left_len)
                max_right = max_token_length - max_left
        else:
            max_left = max_token_length
        if self.truncation_strategy == 'post':
            tokens = tokens[:max_left]
            if pair_tokens is not None:
                pair_tokens = pair_tokens[:max_right]
        elif self.truncation_strategy == 'pre':
            tokens = tokens[-max_left:]
            if pair_tokens is not None:
                pair_tokens = pair_tokens[-max_right:]
        return tokens, pair_tokens

    def raw_tokenizer(self) -> object:
        """ Return raw tokenizer, i.e. object of `tokenizers.BertWordPieceTokenizer` or `sentencepiece.SentencePieceProcessor`
        """
        return self._tokenizer

    @abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_vocab_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def id_to_token(self, idx: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_vocab(self) -> Dict:
        raise NotImplementedError


class SPTokenizer(Tokenizer):
    """ SentencePiece Tokenizer
    Wrap for `sentencepiece`.
    """
    def __init__(self, vocab_path: str, lowercase: bool = False):
        """
        Args:
          - vocab_path: str, path to vocab
          - lowercase: bool, whether to do lowercase, default False
        """
        super().__init__(vocab_path, lowercase=lowercase)
        self._tokenizer = SentencePieceProcessor()
        self._tokenizer.Load(self.vocab_path)

        self.special_tokens.PAD = self._tokenizer.id_to_piece(self._tokenizer.pad_id())
        self.special_tokens.UNK = self._tokenizer.id_to_piece(self._tokenizer.unk_id())

    def get_vocab_size(self) -> int:
        """ Return vocab size
        """
        return self._tokenizer.get_piece_size()

    def token_to_id(self, token: str) -> int:
        """ Convert the input token to corresponding index
        Args:
          - token: str
        Return:
          int
        """
        return self._tokenizer.piece_to_id(token)

    def id_to_token(self, idx: int) -> str:
        """ Convert index to corresponding token
        Args:
          - idx: int
        Return:
          str
        """
        if idx < self.get_vocab_size():
            return self._tokenizer.id_to_piece(idx)
        return ''

    def tokenize(self, sequence: str) -> List[str]:
        """ Tokenize sequence to token peices.
        Args:
          - sequence: str
        Return:
          List[str]
        """
        return self._tokenizer.encode_as_pieces(sequence)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> List[str]:
        """ Decode indexs to tokens
        Args:
          - ids: List[int]
          - skip_special_tokens: bool, optioanl, whether to skip special tokens, default `True`
        Return:
          List[str]
        """
        tokens = [self.id_to_token(idx) for idx in ids]
        if skip_special_tokens:
            return [token for token in tokens if token not in self.special_tokens]
        return tokens

    def get_vocab(self) -> Dict:
        """ Return vocabulary
        """
        return {self._tokenizer.id_to_piece(idx): idx for idx in range(self.get_vocab_size())}


class WPTokenizer(Tokenizer):
    """ WordPieceTokenizer
    Wrap for `BertWordPieceTokenizer`.
    """
    def __init__(self, vocab_path: str, lowercase: bool = False):
        """
        Args:
          - vocab_path: str, path to vocab
          - lowercase: bool, whether to do lowercase, default False
        """
        super().__init__(vocab_path, lowercase=lowercase)
        self._tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=lowercase)

    def get_vocab_size(self) -> int:
        """ Return vocab size
        """
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        """ Convert the input token to corresponding index
        Args:
          - token: str
        Return:
          int
        """
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, idx: int) -> str:
        """ Convert index to corresponding token
        Args:
          - idx: int
        Return:
          str
        """
        if idx < self.get_vocab_size():
            return self._tokenizer.id_to_token(idx)
        return ''

    def tokenize(self, sequence: str) -> List[str]:
        """ Tokenize sequence to token peices.
        Args:
          - sequence: str
        Return:
          List[str]
        """
        encoded = self._tokenizer.encode(sequence)
        return encoded.tokens[1:-1]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> List[str]:
        """ Decode indexs to tokens
        Args:
          - ids: List[int]
          - skip_special_tokens: bool, optioanl, whether to skip special tokens, default `True`
        Return:
          List[str]
        """
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).split()

    def get_vocab(self) -> Dict:
        """ Return vocabulary
        """
        return self._tokenizer.get_vocab()

    def add_special_tokens(self, tokens: List[str]):
        """ Specify special tokens, the tokenizer will reserve special tokens as a whole (i.e. don't split them) in tokenizing.
        Currently, only the WPTokenizer supports specifying special tokens.
        Args:
          - tokens: List[str], special tokens
        """
        self._tokenizer.add_special_tokens(tokens)
