# -*- coding: utf-8 -*-

import os
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
from langml.baselines.contrastive.utils import whitespace_tokenize
from langml.baselines.contrastive.simcse import SimCSE, DataLoader, TFDataLoader


@click.group()
def contrastive():
    """contrastive learning command line tools"""
    pass


@contrastive.command()
@click.option('--backbone', type=str, default='bert',
              help='specify backbone: bert | roberta | albert')
@click.option('--epoch', type=int, default=1, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=2e-5, help='learning rate')
@click.option('--dropout_rate', type=float, default=0.1, help='dropout rate')
@click.option('--temperature', type=float, default=5e-2, help='temperature')
@click.option('--pooling_strategy', type=str, default='cls',
              help='specify pooling_strategy from ["cls", "first-last-avg", "last-avg"]')
@click.option('--max_len', type=int, default=512, help='max len')
@click.option('--early_stop', type=int, default=1, help='patience of early stop')
@click.option('--monitor', type=str, default='loss', help='metrics monitor')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--config_path', type=str, required=True, help='bert config path')
@click.option('--ckpt_path', type=str, required=True, help='bert checkpoint path')
@click.option('--vocab_path', type=str, required=True, help='bert vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--test_path', type=str, required=False, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
@click.option('--apply_aeda', is_flag=True, default=False, help='apply AEDA to augment data')
@click.option('--aeda_language', type=str, required=False, default=None, help='specify AEDA language, ["EN", "CN"]')
@click.option('--do_evaluate', is_flag=True, default=False, help='do evaluation')
@click.option('--distributed_training', is_flag=True, default=False, help='distributed training')
@click.option('--distributed_strategy', type=str, default='MirroredStrategy', help='distributed training strategy')
def simcse(backbone: str, epoch: int, batch_size: int, learning_rate: float, dropout_rate: float,
           temperature: float, pooling_strategy: str, max_len: Optional[int], early_stop: int,
           monitor: str, lowercase: bool, tokenizer_type: Optional[str], config_path: str,
           ckpt_path: str, vocab_path: str, train_path: str, test_path: str, save_dir: str,
           verbose: int, apply_aeda: bool, aeda_language: str, do_evaluate: bool,
           distributed_training: bool, distributed_strategy: str):

    params = Parameters()
    params.add('learning_rate', learning_rate)
    params.add('dropout_rate', dropout_rate)
    params.add('temperature', temperature)
    model_instance = SimCSE(config_path, ckpt_path, params, backbone=backbone)

    # check distribute
    if distributed_training:
        assert TF_KERAS, 'Please `export TF_KERAS=1` to support distributed training!'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    aeda_tokenize = whitespace_tokenize
    if apply_aeda:
        assert aeda_language is not None, 'please specify aeda_language when specify --apply_aeda'
        if aeda_language == 'CN':
            try:
                import jieba
            except ImportError:
                raise ValueError('In order to apply AEDA for chinese data, '
                                 'please run `pip install jieba` to install jieba package')
            aeda_tokenize = jieba.lcut

    train_data, _ = DataLoader.load_data(
        train_path,
        apply_aeda=apply_aeda,
        aeda_tokenize=aeda_tokenize,
        aeda_language=aeda_language,
    )
    info(f'train data size: {len(train_data)}')
    if test_path is not None:
        _, test_data_with_label = DataLoader.load_data(
            test_path,
            apply_aeda=False,
        )
        info(f'test data size: {len(test_data_with_label)}')
    else:
        test_data_with_label = []

    # set tokenizer
    if tokenizer_type == 'wordpiece':
        tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
    elif tokenizer_type == 'sentencepiece':
        tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
    else:
        # auto deduce
        tokenizer = auto_tokenizer(vocab_path, lowercase=lowercase)
    tokenizer.enable_truncation(max_length=max_len)

    if distributed_training:
        strategy = getattr(tf.distribute, distributed_strategy)()
        with strategy.scope():
            model, encoder = model_instance.build_model(pooling_strategy=pooling_strategy, lazy_restore=True)
    else:
        model, encoder = model_instance.build_model(pooling_strategy=pooling_strategy)

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
    else:
        train_dataloader = DataLoader(train_data, tokenizer, batch_size=batch_size)
    train_dataset = train_dataloader(random=True)

    model.fit(train_dataset,
              steps_per_epoch=len(train_dataloader),
              verbose=verbose,
              epochs=epoch,
              callbacks=[early_stop_callback, save_checkpoint_callback])

    # clear model
    del model
    if distributed_training:
        del strategy
    K.clear_session()
    # restore best model
    model, encoder = model_instance.build_model(pooling_strategy=pooling_strategy)
    model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # save model
    info('start to save frozen')
    save_frozen(encoder, os.path.join(save_dir, 'frozen_encoder_model'))
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, os.path.basename(vocab_path)))
    # compute corrcoef
    if do_evaluate and test_data_with_label:
        info('done to training! start to compute metrics...')
        evaluator = SpearmanEvaluator(encoder, tokenizer)
        info(f'test corrcoef: {evaluator.compute_corrcoef(test_data_with_label)}')
