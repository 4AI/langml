# -*- coding: utf-8 -*-

import json
from typing import Callable, Optional, Tuple, Union

import numpy as np

from langml.tensor_typing import Models
from langml.plm.bert import BERT
from langml.utils import load_variables


def load_albert(config_path: str,
                checkpoint_path: str,
                seq_len: Optional[int] = None,
                pretraining: bool = False,
                with_mlm: bool = True,
                with_nsp: bool = True,
                lazy_restore: bool = False,
                weight_prefix: Optional[str] = None,
                dropout_rate: float = 0.0,
                **kwargs) -> Union[Tuple[Models, Callable], Tuple[Models, Callable, Callable]]:
    """ Load pretrained ALBERT
    Args:
      - config_path: str, path of albert config
      - checkpoint_path: str, path of albert checkpoint
      - seq_len: Optional[int], specify fixed input sequence length, default None
      - pretraining: bool, pretraining mode, default False
      - with_mlm: bool, whether to use mlm task in pretraining, default True
      - with_nsp: bool, whether to use nsp/sop task in pretraining, default True
      - lazy_restore: bool, whether to restore pretrained weights lazily, default False.
        Set it as True for distributed training.
      - weight_prefix: Optional[str], prefix name of weights, default None.
        You can set a prefix name in unshared siamese networks.
       - dropout_rate: float, dropout rate, default 0.
    Return:
      - model: keras model
      - bert: bert instance
      - restore: conditionally, it will return when lazy_restore=True
    """
    # initialize model from config
    with open(config_path, 'r') as reader:
        config = json.load(reader)
    if seq_len is not None:
        config['max_position_embeddings'] = min(seq_len, config['max_position_embeddings'])

    bert = BERT(
        config['vocab_size'],
        position_size=config['max_position_embeddings'],
        seq_len=seq_len,
        embedding_dim=config.get('embedding_size') or config.get('hidden_size'),
        hidden_dim=config.get('hidden_size'),
        transformer_blocks=config['num_hidden_layers'],
        attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        feed_forward_activation=config['hidden_act'],
        initializer_range=config['initializer_range'],
        dropout_rate=dropout_rate or config.get('hidden_dropout_prob', 0.0),
        pretraining=pretraining,
        share_weights=True,
        weight_prefix=weight_prefix,
        **kwargs)
    bert.build()
    model = bert(with_mlm=with_mlm, with_nsp=with_nsp)

    def restore(model):
        variables = load_variables(checkpoint_path)
        model.get_layer(name=bert.get_weight_name('Embedding-Token')).set_weights([
            variables('bert/embeddings/word_embeddings'),
        ])
        model.get_layer(name=bert.get_weight_name('Embedding-Position')).set_weights([
            variables('bert/embeddings/position_embeddings')[:config['max_position_embeddings'], :],
        ])
        model.get_layer(name=bert.get_weight_name('Embedding-Segment')).set_weights([
            variables('bert/embeddings/token_type_embeddings'),
        ])
        model.get_layer(name=bert.get_weight_name('Embedding-Norm')).set_weights([
            variables('bert/embeddings/LayerNorm/gamma'),
            variables('bert/embeddings/LayerNorm/beta'),
        ])
        model.get_layer(name=bert.get_weight_name('Embedding-Mapping')).set_weights([
            variables('bert/encoder/embedding_hidden_mapping_in/kernel'),
            variables('bert/encoder/embedding_hidden_mapping_in/bias'),
        ])
        # 以下权重共享
        model.get_layer(name=bert.get_weight_name('Transformer-MultiHeadSelfAttention')).set_weights([
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel'),
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel'),
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel'),
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel'),
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias'),
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias'),
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias'),
            variables('bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias'),
        ])
        model.get_layer(name=bert.get_weight_name('Transformer-MultiHeadSelfAttention-Norm')).set_weights([
            variables('bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma'),
            variables('bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta'),
        ])
        model.get_layer(name=bert.get_weight_name('Transformer-FeedForward')).set_weights([
            variables('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel'),
            variables('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel'),
            variables('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias'),
            variables('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias'),
        ])
        model.get_layer(name=bert.get_weight_name('Transformer-FeedForward-Norm')).set_weights([
            variables('bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma'),
            variables('bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta'),
        ])
        if pretraining:
            if with_mlm:
                model.get_layer(name=bert.get_weight_name('MLM-Dense')).set_weights([
                    variables('cls/predictions/transform/dense/kernel'),
                    variables('cls/predictions/transform/dense/bias'),
                ])
                model.get_layer(name=bert.get_weight_name('MLM-Norm')).set_weights([
                    variables('cls/predictions/transform/LayerNorm/gamma'),
                    variables('cls/predictions/transform/LayerNorm/beta'),
                ])
                model.get_layer(name=bert.get_weight_name('MLM-Match')).set_weights([
                    variables('cls/predictions/output_bias'),
                ])
            if with_nsp:
                model.get_layer(name=bert.get_weight_name('NSP-Dense')).set_weights([
                    variables('bert/pooler/dense/kernel'),
                    variables('bert/pooler/dense/bias'),
                ])
                model.get_layer(name=bert.get_weight_name('NSP')).set_weights([
                    np.transpose(variables('cls/seq_relationship/output_weights')),
                    variables('cls/seq_relationship/output_bias'),
                ])
        return model

    if lazy_restore:
        return model, bert, restore
    model = restore(model)
    return model, bert
