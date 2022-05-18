# -*- coding: utf-8 -*-

import json
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as L
else:
    import keras
    import keras.layers as L

from langml.layers import LayerNorm
from langml.activations import gelu
from langml.transformer.encoder import TransformerEncoderBlock
from langml.tensor_typing import Activation, Tensors, Models
from langml.plm import TokenEmbedding, EmbeddingMatching, Masked
from langml.layers import AbsolutePositionEmbedding
from langml.utils import load_variables


class BERT:
    def __init__(self,
                 vocab_size: int,
                 position_size: int = 512,
                 seq_len: int = 512,
                 embedding_dim: int = 768,
                 hidden_dim: Optional[int] = None,
                 transformer_blocks: int = 12,
                 attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 dropout_rate: float = 0.1,
                 attention_activation: Activation = None,
                 feed_forward_activation: Activation = 'gelu',
                 initializer_range: float = 0.02,
                 pretraining: bool = False,
                 trainable_prefixs: Optional[List] = None,
                 share_weights: bool = False,
                 weight_prefix: Optional[str] = None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.position_size = position_size
        self.embedding_dim = embedding_dim
        self.transformer_blocks = transformer_blocks
        self.attention_heads = attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        if self.attention_activation == 'gelu':
            self.attention_activation = gelu
        if self.feed_forward_activation == 'gelu':
            self.feed_forward_activation = gelu
        self.pretraining = pretraining
        self.trainable_prefixs = trainable_prefixs
        if self.trainable_prefixs is None:
            self.trainable = True
        else:
            self.trainable = False
        self.share_weights = share_weights
        self.weight_prefix = weight_prefix
        self.initializer = keras.initializers.TruncatedNormal(stddev=initializer_range)
        self.is_embedding_mapping = self.hidden_dim is not None and self.embedding_dim != self.hidden_dim

    def get_weight_name(self, name: str) -> str:
        if self.weight_prefix is not None:
            return f'{self.weight_prefix}-{name}'
        return name

    def build(self):
        # emedding layers
        self.token_embedding_layer = TokenEmbedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            trainable=self.trainable,
            embeddings_initializer=self.initializer,
            name=self.get_weight_name('Embedding-Token'),
        )
        self.segment_embedding_layer = L.Embedding(
            input_dim=2,
            output_dim=self.embedding_dim,
            trainable=self.trainable,
            embeddings_initializer=self.initializer,
            name=self.get_weight_name('Embedding-Segment')
        )
        self.add_embedding_layer = L.Add(name=self.get_weight_name('Embedding-Token-Segment'))
        self.position_embedding_layer = AbsolutePositionEmbedding(
            input_dim=self.position_size,
            output_dim=self.embedding_dim,
            mode='add',
            trainable=self.trainable,
            embeddings_initializer=self.initializer,
            name=self.get_weight_name('Embedding-Position'),
        )
        # layernorm
        self.embedding_norm_layer = LayerNorm(
            trainable=self.trainable,
            name=self.get_weight_name('Embedding-Norm'),
        )
        # dropout
        self.embedding_dropout_layer = L.Dropout(
            self.dropout_rate,
            name=self.get_weight_name('Embedding-Dropout'),
        )
        # embedding mapping
        if self.is_embedding_mapping:
            self.embedding_mapping_layer = L.Dense(
                self.hidden_dim,
                kernel_initializer=self.initializer,
                name=self.get_weight_name('Embedding-Mapping')
            )
        # transformer
        self.transformer_layer = TransformerEncoderBlock(
            blocks=self.transformer_blocks,
            attention_heads=self.attention_heads,
            hidden_dim=self.intermediate_size,
            attention_activation=self.attention_activation,
            feed_forward_activation=self.feed_forward_activation,
            dropout_rate=self.dropout_rate,
            name=self.get_weight_name('Transformer'),
            share_weights=self.share_weights
        )

    def get_inputs(self) -> List[Tensors]:
        # Input Placeholder
        t_in = L.Input(shape=(self.seq_len, ), name=self.get_weight_name('Input-Token'))
        s_in = L.Input(shape=(self.seq_len, ), name=self.get_weight_name('Input-Segment'))
        m_in = L.Input(shape=(self.seq_len, ), name=self.get_weight_name('Input-Masked'))
        return [t_in, s_in, m_in]

    def get_embedding(self, inputs: List[Tensors]) -> List[Tensors]:
        token_embedding, embedding_weights = self.token_embedding_layer(inputs[0])
        segment_embedding = self.segment_embedding_layer(inputs[1])
        token_segment_embedding = self.add_embedding_layer([token_embedding, segment_embedding])
        embedding = self.position_embedding_layer(token_segment_embedding)
        return [embedding, embedding_weights]

    def is_trainable(self, layer: L.Layer) -> bool:
        if isinstance(self.trainable_prefixs, (list, tuple, set)):
            if any(layer.name.startswith(prefix) for prefix in self.trainable_prefixs):
                return True
            return False
        return self.trainable

    def __call__(self,
                 inputs: Optional[Union[Tuple, List]] = None,
                 return_model: bool = True,
                 with_mlm: bool = True,
                 with_nsp: bool = True,
                 custom_embedding_callback: Optional[Callable] = None) -> Models:
        if inputs is None:
            inputs = self.get_inputs()
        assert isinstance(inputs, (tuple, list)) and len(inputs) > 1, '`inputs` should be a tuple/list consisting of placeholders and stores token, segment, and masked placeholders respectively.  Note that the masked placeholder is optional for finetuning.' # NOQA
        # embedding
        if custom_embedding_callback is not None:
            embedding, embedding_weights = custom_embedding_callback(inputs)
        else:
            embedding, embedding_weights = self.get_embedding(inputs)
        x = self.embedding_norm_layer(embedding)
        x = self.embedding_dropout_layer(x)
        if self.is_embedding_mapping:
            x = self.embedding_mapping_layer(x)
        # transformer
        x = self.transformer_layer(x)
        if self.pretraining:
            # pretrain
            # don't support parameter sharing for the pretraining phase.
            assert with_mlm or with_nsp, '`with_mlm` and `with_nsp` cannot be `False` at the same time'
            if with_mlm:
                xi = L.Dense(
                    units=self.embedding_dim,
                    activation=self.feed_forward_activation,
                    name=self.get_weight_name('MLM-Dense')
                )(x)
                xi = LayerNorm(name=self.get_weight_name('MLM-Norm'))(xi)
                xi = EmbeddingMatching(name=self.get_weight_name('MLM-Match'))([xi, embedding_weights])
                mask_output = Masked(name=self.get_weight_name('MLM'))([xi, inputs[-1]])
            if with_nsp:
                xi = L.Lambda(lambda t: t[:, 0], name=self.get_weight_name('cls'))(x)
                xi = L.Dense(
                    units=self.hidden_dim or self.embedding_dim,
                    activation='tanh',
                    name=self.get_weight_name('NSP-Dense'),
                )(xi)
                nsp_output = L.Dense(
                    units=2,
                    activation='softmax',
                    name=self.get_weight_name('NSP'),
                )(xi)
            outputs = []
            if with_mlm:
                outputs.append(mask_output)
            if with_nsp:
                outputs.append(nsp_output)
            if return_model:
                model = keras.models.Model(inputs=inputs, outputs=outputs)
                for layer in model.layers:
                    layer.trainable = self.is_trainable(layer)
                return model
            return outputs
        else:
            # finetune
            inputs = inputs[:2]
            if return_model:
                model = keras.models.Model(inputs=inputs, outputs=x)
                for layer in model.layers:
                    layer.trainable = self.is_trainable(layer)

                return model
            return x


def load_bert(config_path: str,
              checkpoint_path: str,
              seq_len: Optional[int] = None,
              pretraining: bool = False,
              with_mlm: bool = True,
              with_nsp: bool = True,
              lazy_restore: bool = False,
              weight_prefix: Optional[str] = None,
              dropout_rate: float = 0.0,
              **kwargs) -> Union[Tuple[Models, Callable], Tuple[Models, Callable, Callable]]:
    """ Load pretrained BERT/RoBERTa
    Args:
      - config_path: str, path of albert config
      - checkpoint_path: str, path of albert checkpoint
      - seq_len: Optional[int], specify fixed input sequence length, default None
      - pretraining: bool, pretraining mode, default False
      - with_mlm: bool, whether to use mlm task in pretraining, default True
      - with_nsp: bool, whether to use nsp task in pretraining, default True
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
        weight_prefix=weight_prefix,
        **kwargs)
    bert.build()
    model = bert(with_mlm=with_mlm, with_nsp=with_nsp)

    def restore(model):
        # restore weights
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
        try:
            # BERT 并没有这一层
            model.get_layer(name=bert.get_weight_name('Embedding-Mapping')).set_weights([
                variables('bert/encoder/embedding_hidden_mapping_in/kernel'),
                variables('bert/encoder/embedding_hidden_mapping_in/bias'),
            ])
        except ValueError:
            print('Skip Embedding-Mapping')
            pass
        for i in range(config['num_hidden_layers']):
            model.get_layer(name=bert.get_weight_name('Transformer-%d-MultiHeadSelfAttention' % i)).set_weights([
                variables('bert/encoder/layer_%d/attention/self/query/kernel' % i),
                variables('bert/encoder/layer_%d/attention/self/key/kernel' % i),
                variables('bert/encoder/layer_%d/attention/self/value/kernel' % i),
                variables('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
                variables('bert/encoder/layer_%d/attention/self/query/bias' % i),
                variables('bert/encoder/layer_%d/attention/self/key/bias' % i),
                variables('bert/encoder/layer_%d/attention/self/value/bias' % i),
                variables('bert/encoder/layer_%d/attention/output/dense/bias' % i),
            ])
            model.get_layer(name=bert.get_weight_name(
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i)
            ).set_weights([
                variables('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
                variables('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
            ])
            model.get_layer(name=bert.get_weight_name('Transformer-%d-FeedForward' % i)).set_weights([
                variables('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
                variables('bert/encoder/layer_%d/output/dense/kernel' % i),
                variables('bert/encoder/layer_%d/intermediate/dense/bias' % i),
                variables('bert/encoder/layer_%d/output/dense/bias' % i),
            ])
            model.get_layer(name=bert.get_weight_name('Transformer-%d-FeedForward-Norm' % i)).set_weights([
                variables('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
                variables('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
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
