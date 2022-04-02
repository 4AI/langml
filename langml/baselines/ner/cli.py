# -*- coding: utf-8 -*-

import os
import json
from typing import Optional
from shutil import copyfile

import click
import tensorflow as tf
from langml import TF_KERAS
from langml.baselines.ner.bert_crf import BertCRF
from langml.baselines.ner.lstm_crf import LSTMCRF
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
else:
    import keras
    import keras.backend as K
from langml.log import info
from langml.tokenizer import WPTokenizer, SPTokenizer
from langml.baselines import Parameters
from langml.baselines.ner import compute_detail_metrics
from langml.baselines.ner.dataloader import DataLoader, TFDataLoader
from langml.model import save_frozen
from langml.utils import auto_tokenizer


def train(
    model_instance: object, params: Parameters, epoch: int, save_dir: str,
    train_path: str, dev_path: str, test_path: str, vocab_path: str,
    tokenizer_type: str, lowercase: bool, max_len: int, batch_size: int,
    distributed_training: bool, distributed_strategy: str,
    monitor: str, early_stop: int, verbose: int,
):
    # check distribute
    if distributed_training:
        assert TF_KERAS, 'Please `export TF_KERAS=1` to support distributed training!'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    is_bert = isinstance(model_instance, BertCRF)

    train_datas, label2id = DataLoader.load_data(train_path, build_vocab=True)
    id2label = {v: k for k, v in label2id.items()}
    dev_datas = DataLoader.load_data(dev_path)
    test_datas = None
    if test_path is not None:
        test_datas = DataLoader.load_data(test_path)
    info(f'labels: {label2id}')
    info(f'train size: {len(train_datas)}')
    info(f'valid size: {len(dev_datas)}')
    if test_path is not None:
        info(f'test size: {len(test_datas)}')

    # set tokenizer
    if tokenizer_type == 'wordpiece':
        tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
    elif tokenizer_type == 'sentencepiece':
        tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
    else:
        # auto deduce
        tokenizer = auto_tokenizer(vocab_path, lowercase=lowercase)
    tokenizer.enable_truncation(max_length=max_len)

    # set params
    params.add('tag_size', len(label2id))
    params.add('vocab_size', tokenizer.get_vocab_size())

    if distributed_training:
        strategy = getattr(tf.distribute, distributed_strategy)()
        with strategy.scope():
            if is_bert:
                model = model_instance.build_model(lazy_restore=True)
            else:
                model = model_instance.build_model()
    else:
        model = model_instance.build_model()

    if distributed_training:
        info('distributed training! using `TFDataLoader`')
        train_DataLoader = TFDataLoader(train_datas, tokenizer, label2id,
                                        batch_size=batch_size, max_len=max_len, is_bert=is_bert)
        dev_DataLoader = TFDataLoader(dev_datas, tokenizer, label2id,
                                      batch_size=batch_size, max_len=max_len, is_bert=is_bert)
    else:
        train_DataLoader = DataLoader(train_datas, tokenizer, label2id,
                                      batch_size=batch_size, max_len=max_len, is_bert=is_bert)
        dev_DataLoader = DataLoader(dev_datas, tokenizer, label2id,
                                    batch_size=batch_size, max_len=max_len, is_bert=is_bert)
    train_dataset = train_DataLoader(random=True)
    dev_dataset = dev_DataLoader(random=False)

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stop,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    save_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, 'best_model.weights'),
        save_best_only=True,
        save_weights_only=True,
        monitor=monitor,
        mode='auto')
    model.fit(train_dataset,
              steps_per_epoch=len(train_DataLoader),
              validation_data=dev_dataset,
              validation_steps=len(dev_DataLoader),
              verbose=verbose,
              epochs=epoch,
              callbacks=[early_stop_callback, save_checkpoint_callback])
    # clear model
    del model
    if distributed_training:
        del strategy
    K.clear_session()
    # restore model
    model = model_instance.build_model()
    model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # compute detail metrics
    info('done to training! start to compute detail metrics...')
    print('develop metrics:')
    dev_DataLoader = DataLoader(dev_datas, tokenizer, label2id,
                                batch_size=batch_size, max_len=max_len, is_bert=is_bert)
    compute_detail_metrics(model, dev_DataLoader, id2label, is_bert=is_bert)
    if test_datas:
        print('test metrics:')
        test_DataLoader = DataLoader(test_datas, tokenizer, label2id,
                                     batch_size=batch_size, max_len=max_len, is_bert=is_bert)
        compute_detail_metrics(model, test_DataLoader, id2label, is_bert=is_bert)
    # save model
    info('start to save frozen')
    save_frozen(model, os.path.join(save_dir, 'frozen_model'))
    info('start to save label')
    with open(os.path.join(save_dir, 'label2id.json'), 'w', encoding='utf-8') as writer:
        json.dump(label2id, writer, ensure_ascii=False)
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, os.path.basename(vocab_path)))


@click.group()
def ner():
    """ner command line tools"""
    pass


@ner.command()
@click.option('--backbone', type=str, default='bert',
              help='specify backbone: bert | roberta | albert')
@click.option('--epoch', type=int, default=20, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=2e-5, help='learning rate')
@click.option('--dropout_rate', type=float, default=0.2, help='dropout rate')
@click.option('--max_len', type=int, default=512, help='max len')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--config_path', type=str, required=True, help='bert config path')
@click.option('--ckpt_path', type=str, required=True, help='bert checkpoint path')
@click.option('--vocab_path', type=str, required=True, help='bert vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, help='dev path')
@click.option('--test_path', type=str, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--monitor', type=str, default='val_loss', help='monitor for keras callback')
@click.option('--early_stop', type=int, default=10, help='patience to early stop')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
@click.option('--distributed_training', is_flag=True, default=False, help='distributed training')
@click.option('--distributed_strategy', type=str, default='MirroredStrategy', help='distributed training strategy')
def bert_crf(backbone: str, epoch: int, batch_size: int, learning_rate: float,
             dropout_rate: float, max_len: Optional[int], lowercase: bool,
             tokenizer_type: Optional[str], config_path: str, ckpt_path: str,
             vocab_path: str, train_path: str, dev_path: str, test_path: str,
             save_dir: str, monitor: str, early_stop: int, verbose: int,
             distributed_training: bool, distributed_strategy: str):

    params = Parameters({
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
    })

    model_instance = BertCRF(config_path, ckpt_path, params, backbone=backbone)
    train(
        model_instance, params, epoch, save_dir,
        train_path, dev_path, test_path, vocab_path,
        tokenizer_type, lowercase, max_len, batch_size,
        distributed_training, distributed_strategy,
        monitor, early_stop, verbose
    )


@ner.command()
@click.option('--epoch', type=int, default=20, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=1e-3, help='learning rate')
@click.option('--dropout_rate', type=float, default=0.2, help='dropout rate')
@click.option('--embedding_size', type=int, default=200, help='embedding size')
@click.option('--hidden_size', type=int, default=128, help='hidden size')
@click.option('--max_len', type=int, default=None, help='max len')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--vocab_path', type=str, required=True, help='vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, help='dev path')
@click.option('--test_path', type=str, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--monitor', type=str, default='val_loss', help='monitor for keras callback')
@click.option('--early_stop', type=int, default=10, help='patience to early stop')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
@click.option('--distributed_training', is_flag=True, default=False, help='distributed training')
@click.option('--distributed_strategy', type=str, default='MirroredStrategy', help='distributed training strategy')
def lstm_crf(epoch: int, batch_size: int, learning_rate: float, dropout_rate: float,
             embedding_size: int, hidden_size: int, max_len: Optional[int],
             lowercase: bool, tokenizer_type: Optional[str], vocab_path: str,
             train_path: str, dev_path: str, test_path: str, save_dir: str,
             monitor: str, early_stop: int, verbose: int, distributed_training: bool,
             distributed_strategy: str):

    params = Parameters({
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'embedding_size': embedding_size,
        'hidden_size': hidden_size
    })

    model_instance = LSTMCRF(params)

    train(
        model_instance, params, epoch, save_dir,
        train_path, dev_path, test_path, vocab_path,
        tokenizer_type, lowercase, max_len, batch_size,
        distributed_training, distributed_strategy,
        monitor, early_stop, verbose
    )
