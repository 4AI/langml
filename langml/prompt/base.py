# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from typing import Dict, List

from langml.tensor_typing import Models
from langml.tokenizer import Tokenizer
from langml.plm import load_albert, load_bert
from langml.log import info


class Template:
    def __init__(self, template: List[str], label_tokens_map: Dict[str, List[str]], tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.unk_id = self.tokenizer.token_to_id(self.tokenizer.special_tokens.UNK)

        self.template_ids = self.encode_template(template)
        self.label2tokens, self.id2label = self.encode_label_tokens_map(label_tokens_map)
        info(f'template ids: {self.template_ids}')

    def __len__(self) -> int:
        return len(self.template_ids)

    def encode_template(self, template: str) -> List[int]:
        return [self.tokenizer.token_to_id(token) for token in template]

    def encode_label_tokens_map(self, label_tokens_map: Dict[str, List[str]]) -> Dict[str, List[int]]:
        label2ids, id2label = {}, {}
        for label, tokens in label_tokens_map.items():
            token_ids = []
            for token in tokens:
                token_id = self.tokenizer.token_to_id(token)
                assert token_id != self.unk_id, f'unknown token {token}! please specify a token from vocabulary'
                token_ids.append(token_id)
                id2label[token_id] = label
            label2ids[label] = token_ids
        return label2ids, id2label

    def decode_label(self, idx: int, default='<UNK>') -> str:
        return self.id2label.get(idx, default)


class BasePromptModel(metaclass=ABCMeta):
    def __init__(self,
                 plm_backbone: str,
                 plm_config_path: str,
                 plm_ckpt_path: str,
                 template: Template,
                 learning_rate: float = 1e-5,
                 freeze_plm: bool = True) -> None:
        """ Initialize Prompt Model
        Args:
          - plm_backbone: str, backbone of pretrained language model
          - plm_config_path: str, configure path of pretrained language model
          - plm_ckpt_path: str, checkpoint path of pretrained language model
          - template: List[str], template
          - label_tokens_map: str, verbalizer, map of label to tokens
          - tokenizer: langml.Tokenizer, tokenizer
          - learning_rate: float, learning rate
          - freeze_plm: bool, whether to freeze pretrained language model weights
        """
        self.model = None
        self.freeze_plm = freeze_plm
        if plm_backbone == 'albert':
            _, self.plm, self.lazy_restore_callback = load_albert(
                config_path=plm_config_path,
                checkpoint_path=plm_ckpt_path,
                pretraining=True,
                with_mlm=True,
                with_nsp=False,
                lazy_restore=True)
        else:
            _, self.plm, self.lazy_restore_callback = load_bert(
                config_path=plm_config_path,
                checkpoint_path=plm_ckpt_path,
                pretraining=True,
                with_mlm=True,
                with_nsp=False,
                lazy_restore=True)
        self.template = template
        self.learning_rate = learning_rate

    @abstractmethod
    def build_model(self) -> Models:
        raise NotImplementedError


class BasePromptTask(metaclass=ABCMeta):
    def __init__(self, prompt_model: BasePromptModel, tokenizer: Tokenizer) -> None:
        self.prompt_model = prompt_model
        self.template = prompt_model.template
        self.tokenizer = tokenizer
        self.mask_id = self.tokenizer.token_to_id(self.tokenizer.special_tokens.MASK)
        self.model = self.prompt_model.build_model()

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class BaseDataGenerator(metaclass=ABCMeta):
    @abstractmethod
    def make_iter(self, random: bool = False):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __call__(self, random: bool = False):
        while True:
            for inputs, labels in self.make_iter(random=random):
                yield inputs, labels
