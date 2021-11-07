# -*- coding: utf-8 -*-

"""
Prompt-Base finetune.
"""

import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from langml.log import info
from langml.plm.bert import load_bert
from langml.plm.albert import load_albert
from langml import TF_KERAS, TF_VERSION
if TF_KERAS:
    import tensorflow.keras as keras
    from tensorflow.keras.preprocessing.sequence import pad_sequences
else:
    import keras
    from keras.preprocessing.sequence import pad_sequences

from langml.tokenizer import Tokenizer, WPTokenizer, SPTokenizer


re_unused = re.compile(r'\[unused[0-9]+\]')


class DataGenerator:
    def __init__(self,
                 template_ids: List[int],
                 datas: List[str],
                 labels: List[Union[str, List[str]]],
                 tokenizer: Tokenizer,
                 label2id: Dict,
                 mask_id: int,
                 batch_size: int = 16):
        self.batch_size = batch_size
        self.mask_id = mask_id
        self.datas = []
        for data, label in zip(datas, labels):
            tokened = tokenizer.encode(data)
            token_ids = tokened.ids
            token_ids = [token_ids[0]] + template_ids + token_ids[1:-1] + [token_ids[-1]]
            segment_ids = [0] * len(token_ids)
            mask_ids = (np.array(token_ids) == self.mask_id).astype('int')
            if isinstance(label, str):
                label = [label]
            else:
                label = list(label)
            assert len(label) == mask_ids.sum(), 'number of [MASK] should be equal with number of label'
            mask_ids = mask_ids.tolist()
            output_ids = []
            label = label[::-1]
            for token_id, mask_id in zip(token_ids, mask_ids):
                if mask_id == 1:
                    output_ids.append(label2id[label.pop()])
                else:
                    output_ids.append(token_id)
            self.datas.append({
                'token_ids': token_ids,
                'segment_ids': segment_ids,
                'mask_ids': mask_ids,
                'output_ids': output_ids
            })

        self.steps = len(self.datas) // self.batch_size
        if len(self.datas) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random: bool = True):
        idxs = list(range(len(self.datas)))
        if random:
            np.random.shuffle(idxs)
        batch_tokens, batch_segments, batch_mask_ids, batch_outputs = [], [], [], []
        for idx in idxs:
            obj = self.datas[idx]
            batch_tokens.append(obj['token_ids'])
            batch_segments.append(obj['segment_ids'])
            batch_mask_ids.append(obj['mask_ids'])
            batch_outputs.append(obj['output_ids'])

            if len(batch_tokens) == self.batch_size or idx == idxs[-1]:
                batch_outputs = pad_sequences(batch_outputs, truncating='post', padding='post')
                batch_outputs = np.expand_dims(batch_outputs, axis=-1)
                batch_tokens = pad_sequences(batch_tokens, truncating='post', padding='post')
                batch_segments = pad_sequences(batch_segments, truncating='post', padding='post')
                batch_mask_ids = pad_sequences(batch_mask_ids, truncating='post', padding='post')
                yield [batch_tokens, batch_segments, batch_mask_ids], [batch_outputs]
                batch_tokens, batch_segments, batch_mask_ids, batch_outputs = [], [], [], []

    def forfit(self, random: bool = True):
        while True:
            for inputs, labels in self.__iter__(random=random):
                yield inputs, labels


class Prompt:
    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 vocab_path: str,
                 verbalizer: Dict,
                 template: str,
                 backbone: str = 'bert',
                 tokenizer: Optional[Tokenizer] = None,
                 lowercase: bool = False,
                 max_length: int = 512,
                 special_tokens: Optional[List[str]] = None,
                 lazy_restore: bool = False):
        """
        Args:
          - config_path: str, path of PLM config
          - checkpoint_path: str, path of PLM checkpoint
          - vocab_path: str, path of vocabulary
          - verbalizer, Dict, the mapping of tokens to factual labels.
            Please assure vocabulary contain input tokens.
            Strongly recommend using unused tokens as verbalizer tokens.
          - template: str, prompt template
          - backbone: str, optional, specify PLM backbones, options: bert, roberta, albert
          - tokenizer: Optional[Tokenizer], optional, options: SPTokenizer, WPTokenizer
          - lowercase: bool, optional, whether to do lowercase
          - max_length: int, optional, max length of tokenizer, default 512
          - special_tokens: Optional[List[str]], specify special tokens, default None
          - lazy_restore: bool, whether to lazy restore PLMs, set `True` if train distributely.
        """
        self.vocab_path = vocab_path
        self.lowercase = lowercase
        self.lazy_restore_callback = None
        if tokenizer is None:
            if vocab_path.endswith('.txt'):
                info('automatically apply `WPTokenizer`')
                tokenizer = WPTokenizer
            elif vocab_path.endswith('.model'):
                info('automatically apply `SPTokenizer`')
                tokenizer = SPTokenizer
            else:
                raise ValueError("Langml cannot deduce which tokenizer to apply, please assign `tokenizer` manually.")  # NOQA

        self.tokenizer = tokenizer(vocab_path, lowercase=lowercase)
        if special_tokens is not None:
            self.tokenizer.add_special_tokens(special_tokens)
        self.label2id = self._label_map(verbalizer)
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.mask_id = self.tokenizer.token_to_id(self.tokenizer.special_tokens.MASK)

        self.template_ids = self.get_template_ids(template)
        self.tokenizer.enable_truncation(max_length=max_length - len(self.template_ids))

        if backbone == 'albert':
            load_model = load_albert
        elif backbone == 'bert':
            load_model = load_bert
        if lazy_restore:
            self.model, _, self.lazy_restore_callback = load_model(
                config_path, checkpoint_path,
                pretraining=True, with_nsp=False, lazy_restore=True
            )
        else:
            self.model, _ = load_model(
                config_path, checkpoint_path,
                pretraining=True, with_nsp=False,
            )
        self.model.summary()

    def get_template_ids(self, template: str) -> List[int]:
        """ Get template token ids
        Args:
          - template: str
        Return:
          List[int]
        """
        template_ids = []
        start = 0
        for match in re.finditer(r'\[MASK\]', template):
            span = match.span()
            template_ids += self.tokenizer.encode(template[start: span[0]]).ids[1:-1]
            template_ids += [self.mask_id]
            start = span[1]
        if template[start:]:
            template_ids += self.tokenizer.encode(template[start:]).ids[1:-1]
        return template_ids

    def get_available_unused_tokens(self) -> List[Tuple[str, int]]:
        """ Return available unused tokens.
        Strongly recommend using unused tokens as verbalizer tokens.
        """
        unuseds = []
        for token, idx in self.tokenizer.get_vocab().items():
            if re_unused.match(token) and idx not in self.label2id.values():
                unuseds.append((token, idx))
        unuseds.sort(key=lambda x: x[1])
        return unuseds

    def _label_map(self, verbalizer: Dict) -> Dict:
        """ map token to factual labels
        """
        label2id = {}
        unk_id = self.tokenizer.token_to_id(self.tokenizer.special_tokens.UNK)
        for token, label in verbalizer.items():
            token_id = self.tokenizer.token_to_id(token)
            assert token_id is not None and token_id != unk_id, f'the token `{token}` is not found in vocabulary'
            label2id[label] = token_id
        return label2id

    def fit(self,
            datas: List[str],
            labels: List[Union[str, List[str]]],
            epoch: int = 20,
            batch_size: int = 16,
            learning_rate: float = 2e-5,
            do_shuffle: bool = True,
            verbose: int = 1,
            callbacks: Optional[List] = None):
        """ fit model
        Args:
          - datas: List[str], texts
          - labels: List[Union[str, List[str]]], labels
          - epoch: int
          - batch_size: int
          - learning_rate: float
          - do_shuffle: whether to shuffle data in training phase
          - verbose: int, 0 = silent, 1 = progress bar, 2 = one line per epoch
          - callbacks: keras callbacks
        """
        assert len(datas) == len(labels), "datas should have an equal size with labels"
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
        )
        if self.lazy_restore_callback is not None:
            self.lazy_restore_callback(self.model)
        generator = DataGenerator(self.template_ids, datas, labels, self.tokenizer,
                                  self.label2id, mask_id=self.mask_id, batch_size=batch_size)
        info('start to train...')
        self.model.fit(
            generator.forfit(random=do_shuffle),
            steps_per_epoch=len(generator),
            epochs=epoch,
            callbacks=callbacks,
            verbose=verbose
        )

    def predict(self, text: str) -> List:
        """ inference
        Args:
          -  text: str
        """
        tokened = self.tokenizer.encode(text)
        token_ids = tokened.ids
        token_ids = [token_ids[0]] + self.template_ids + token_ids[1:-1] + [token_ids[-1]]
        segment_ids = [0] * len(token_ids)
        token_ids = np.array([token_ids])
        segment_ids = np.array([segment_ids])
        mask_ids = (token_ids == self.mask_id).astype('int')
        if TF_VERSION > 1:
            mask_output = self.model([token_ids, segment_ids, mask_ids])
            mask_output = mask_output[0].numpy()
        else:
            mask_output = self.model.predict([token_ids, segment_ids, mask_ids])
            mask_output = mask_output[0]
        pred = np.argmax(mask_output, axis=1)
        output = pred * mask_ids
        output = output[output > 0].tolist()
        return [self.id2label.get(idx, '<UNK>') for idx in output]

    def save(self, save_path: str):
        self.model.save_weights(save_path)
        info(f'model successfully saved to {save_path}!')

    def load(self, model_path: str):
        self.model.load_weights(model_path)
        info('model successfully loaded!')
