# -*- coding: utf-8 -*-

""" Yet another transformer implementation.
"""

# TODO: Transformer Decoder

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras.layers as L
else:
    import keras.layers as L

from langml.layers import MultiHeadAttention, LayerNorm
from langml.tensor_typing import Tensors, Activation
from langml.transformer import FeedForward
from langml.activations import gelu


class TransformerEncoder:
    def __init__(self,
                 attention_heads: int,
                 hidden_dim: int,
                 attention_activation: Activation = None,
                 feed_forward_activation: Activation = gelu,
                 dropout_rate: float = 0.0,
                 trainable: bool = True,
                 name: str = 'Transformer-Encoder'):
        self.name = name
        self.dropout_rate = dropout_rate
        self.multihead_layer = MultiHeadAttention(head_num=attention_heads,
                                                  return_attention=False,
                                                  attention_activation=attention_activation,
                                                  history_only=False,
                                                  trainable=trainable,
                                                  name=f'{self.name}-MultiHeadSelfAttention')
        if dropout_rate > 0.0:
            self.attn_dropout_layer = L.Dropout(rate=dropout_rate, name=f'{self.name}-MultiHeadSelfAttention-Dropout')
        self.attn_residual_layer = L.Add(name=f'{self.name}-MultiHeadSelfAttention-Add')
        self.attn_layer_norm = LayerNorm(name=f'{self.name}-MultiHeadSelfAttention-Norm', trainable=trainable)
        self.ffn_layer = FeedForward(hidden_dim,
                                     activation=feed_forward_activation,
                                     name=f'{self.name}-FeedForward')
        if dropout_rate > 0.0:
            self.ffn_dropout_layer = L.Dropout(rate=dropout_rate, name=f'{self.name}-FeedForward-Dropout')
        self.ffn_residual_layer = L.Add(name=f'{self.name}-FeedForward-Add')
        self.ffn_layer_norm = LayerNorm(name=f'{self.name}-FeedForward-Norm', trainable=trainable)

    def __call__(self, inputs: Tensors) -> Tensors:
        attn_output = self.multihead_layer(inputs)
        if self.dropout_rate > 0.0:
            attn_output = self.attn_dropout_layer(attn_output)
        if isinstance(inputs, list):
            inputs = inputs[0]
        attn_output = self.attn_residual_layer([inputs, attn_output])
        attn_output = self.attn_layer_norm(attn_output)

        ffn_output = self.ffn_layer(attn_output)
        if self.dropout_rate > 0.0:
            ffn_output = self.ffn_dropout_layer(ffn_output)
        ffn_output = self.ffn_residual_layer([attn_output, ffn_output])
        ffn_output = self.ffn_layer_norm(ffn_output)

        return ffn_output


class TransformerEncoderBlock:
    def __init__(self,
                 blocks: int,
                 attention_heads: int,
                 hidden_dim: int,
                 attention_activation: Activation = None,
                 feed_forward_activation: Activation = gelu,
                 dropout_rate: float = 0.0,
                 trainable: bool = False,
                 name: str = 'TransformerEncoderBlock',
                 share_weights: bool = False):
        if share_weights:
            encoder = TransformerEncoder(attention_heads,
                                         hidden_dim,
                                         attention_activation=attention_activation,
                                         feed_forward_activation=feed_forward_activation,
                                         dropout_rate=dropout_rate,
                                         trainable=trainable,
                                         name=name)
            self.encoders = [encoder for _ in range(blocks)]
        else:
            self.encoders = [
                TransformerEncoder(attention_heads,
                                   hidden_dim,
                                   attention_activation=attention_activation,
                                   feed_forward_activation=feed_forward_activation,
                                   dropout_rate=dropout_rate,
                                   trainable=trainable,
                                   name=f'{name}-{i}')
                for i in range(blocks)
            ]

    def __call__(self, inputs: Tensors) -> Tensors:
        output = inputs
        for encoder in self.encoders:
            output = encoder(output)
        return output
