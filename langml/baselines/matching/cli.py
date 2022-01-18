# -*- coding: utf-8 -*-

import os
import json
from typing import Optional
from shutil import copyfile

import click
import tensorflow as tf
from langml import TF_KERAS, keras, K
from langml.log import info
from langml.baselines import Parameters
from langml.tokenizer import WPTokenizer, SPTokenizer
from langml.model import save_frozen
from langml.utils import auto_tokenizer
from langml.common.evaluator import SpearmanEvaluator
from langml.baselines.matching.sbert import SentenceBert, DataLoader, TFDataLoader


@click.group()
def matching():
    """text matching command line tools"""
    pass


@matching.command()
@click.option('--backbone', type=str, default='bert',
              help='specify backbone: bert | roberta | albert')
@click.option('--epoch', type=int, default=5, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=2e-5, help='learning rate')
@click.option('--dropout_rate', type=float, default=0.1, help='dropout rate')
@click.option('--task', type=str, default='regression',
              help='specify task from ["regression", "classification"]')
@click.option('--pooling_strategy', type=str, default='cls',
              help='specify pooling_strategy from ["cls", "mean", "max"]')
@click.option('--max_len', type=int, default=512, help='max len')
@click.option('--early_stop', type=int, default=1, help='patience of early stop')
@click.option('--monitor', type=str, default='val_loss', help='metrics monitor')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--config_path', type=str, required=True, help='bert config path')
@click.option('--ckpt_path', type=str, required=True, help='bert checkpoint path')
@click.option('--vocab_path', type=str, required=True, help='bert vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, default=None, help='dev path')
@click.option('--test_path', type=str, required=False, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
@click.option('--distributed_training', is_flag=True, default=False, help='distributed training')
@click.option('--distributed_strategy', type=str, default='MirroredStrategy', help='distributed training strategy')
def sbert(backbone: str, epoch: int, batch_size: int, learning_rate: float, dropout_rate: float,
          task: str, pooling_strategy: str, max_len: Optional[int], early_stop: int, monitor: str,
          lowercase: bool, tokenizer_type: Optional[str], config_path: str, ckpt_path: str, vocab_path: str,
          train_path: str, dev_path: str, test_path: str, save_dir: str, verbose: int,
          distributed_training: bool, distributed_strategy: str):

    params = Parameters()
    params.add('learning_rate', learning_rate)
    params.add('dropout_rate', dropout_rate)

    # check distribute
    if distributed_training:
        assert TF_KERAS, 'Please `export TF_KERAS=1` to support distributed training!'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if task == 'classification':
        train_data, label2idx = DataLoader.load_data(train_path, build_vocab=True)
        info(f'label2idx: {label2idx}')
        params.add('tag_size', len(label2idx))
    else:
        train_data = DataLoader.load_data(train_path)
        label2idx = None
    dev_data = DataLoader.load_data(dev_path, label2idx=label2idx)
    info(f'train data size: {len(train_data)}')
    info(f'dev data size: {len(dev_data)}')
    test_data = None
    if test_path is not None:
        test_data = DataLoader.load_data(test_path, label2idx=label2idx)
        info(f'test data size: {len(test_data)}')

    # set tokenizer
    if tokenizer_type == 'wordpiece':
        tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
    elif tokenizer_type == 'sentencepiece':
        tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
    else:
        # auto deduce
        tokenizer = auto_tokenizer(vocab_path, lowercase=lowercase)
    tokenizer.enable_truncation(max_length=max_len)

    model_instance = SentenceBert(config_path, ckpt_path, params, backbone=backbone)

    if distributed_training:
        strategy = getattr(tf.distribute, distributed_strategy)()
        with strategy.scope():
            model, encoder = model_instance.build_model(
                task, pooling_strategy=pooling_strategy, lazy_restore=True)
    else:
        model, encoder = model_instance.build_model(
            task, pooling_strategy=pooling_strategy, lazy_restore=True)

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stop,
        verbose=0,
        mode='auto',
        restore_best_weights=True
    )
    save_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, 'best_model.weights'),
        save_best_only=True,
        save_weights_only=True,
        monitor=monitor,
        mode='auto')

    if distributed_training:
        info('distributed training! using `TFDataLoader`')
        train_dataloader = TFDataLoader(train_data, tokenizer, batch_size=batch_size)
        dev_dataloader = TFDataLoader(dev_data, tokenizer, batch_size=batch_size)
    else:
        train_dataloader = DataLoader(train_data, tokenizer, batch_size=batch_size)
        dev_dataloader = DataLoader(dev_data, tokenizer, batch_size=batch_size)
    train_dataset = train_dataloader(random=True)
    dev_dataset = train_dataloader(random=False)

    model.fit(train_dataset,
              steps_per_epoch=len(train_dataloader),
              validation_data=dev_dataset,
              validation_steps=len(dev_dataloader),
              verbose=verbose,
              epochs=epoch,
              callbacks=[early_stop_callback, save_checkpoint_callback])

    # clear model
    del model
    if distributed_training:
        del strategy
    K.clear_session()
    # restore best model
    model, encoder = model_instance.build_model(task, pooling_strategy=pooling_strategy)
    model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # save model
    info('start to save frozen')
    save_frozen(encoder, os.path.join(save_dir, 'frozen_encoder_model'))
    # save bert vocab
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, os.path.basename(vocab_path)))
    if task == 'classification':
        # save label2idx
        with open(os.path.join(save_dir, 'label2idx.json'), 'w') as writer:
            json.dump(label2idx, writer, ensure_ascii=False)
    # compute corrcoef
    if test_data is not None:
        info('done to training! start to compute metrics...')
        test_dataloader = DataLoader(test_data, tokenizer, batch_size=batch_size)
        test_dataset = test_dataloader(random=False)
        loss, accuracy = model.evaluate(test_dataset, steps=len(test_dataloader), verbose=verbose)
        info(f'test loss: {loss}, test accuracy: {accuracy}')
        evaluator = SpearmanEvaluator(encoder, tokenizer)
        info(f'test corrcoef: {evaluator.compute_corrcoef(test_data)}')
