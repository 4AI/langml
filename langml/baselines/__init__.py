# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Optional


class BaselineModel(metaclass=ABCMeta):
    @abstractmethod
    def build_model(self, *args, **kwargs):
        raise NotImplementedError


class BaseDataLoader(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def load_data():
        raise NotImplementedError

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


class Parameters:
    """ Hyper-Parameters
    """
    def __init__(self, data: Optional[Dict] = None):
        if data is not None:
            for name, value in data.items():
                setattr(self, name, self._wrap(value))

    def _wrap(self, value: Any):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Parameters(value) if isinstance(value, dict) else value

    def add(self, name, value):
        setattr(self, name, value)
