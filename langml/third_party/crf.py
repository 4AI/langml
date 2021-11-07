# -*- coding: utf-8 -*-

# This code implements basic operations of CRF
# Modified from https://github.com/tensorflow/addons (compatible with keras, tf.keras)

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from typing import Optional, Union, List, Tuple

from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
else:
    import keras
    import keras.backend as K

import numpy as np
import tensorflow as tf
from typeguard import typechecked

from langml.tensor_typing import Tensors


def viterbi_decode(score: Tensors, trans: Tensors) -> Tuple[Tensors, Tensors]:
    """
    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      trans: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + trans
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if inputs is not None:
        batch_size = K.shape(inputs)[0]
        dtype = K.shape(inputs)

    return K.zeros(shape=(batch_size, cell.state_size), dtype=dtype)


def crf_filtered_inputs(inputs: Tensors, tag_bitmap: Tensors) -> tf.Tensor:
    """Constrains the inputs to filter out certain tags at each time step.
    tag_bitmap limits the allowed tags at each input time step.
    This is useful when an observed output at a given time step needs to be
    constrained to a selected set of tags.
    Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
        representing all active tags at each index for which to calculate the
        unnormalized score.
    Returns:
    filtered_inputs: A [batch_size] vector of unnormalized sequence scores.
    """

    # set scores of filtered out inputs to be -inf.
    filtered_inputs = tf.where(
        tag_bitmap,
        inputs,
        tf.fill(K.shape(inputs), K.cast(float("-inf"), K.dtype(inputs))),
    )
    return filtered_inputs


def crf_sequence_score(
    inputs: Tensors,
    tag_indices: Tensors,
    sequence_lengths: Tensors,
    transition_params: Tensors,
) -> tf.Tensor:
    """Computes the unnormalized score for a tag sequence.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    tag_indices = K.cast(tag_indices, dtype='int32')
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')

    # If max_seq_len is 1, we skip the score calculation and simply gather the
    # unary potentials of the single tag.
    def _single_seq_fn():
        batch_inds = K.reshape(K.arange(0, K.shape(inputs)[0]), [-1, 1])
        indices = K.concatenate([batch_inds, tf.zeros_like(batch_inds)], axis=1)

        tag_inds = tf.gather_nd(tag_indices, indices)
        tag_inds = K.reshape(tag_inds, [-1, 1])
        indices = K.concatenate([indices, tag_inds], axis=1)

        sequence_scores = tf.gather_nd(inputs, indices)

        sequence_scores = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(sequence_scores),
            sequence_scores,
        )
        return sequence_scores

    def _multi_seq_fn():
        # Compute the scores of the given tag sequence.
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(
            tag_indices, sequence_lengths, transition_params
        )
        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    return K.switch(K.equal(K.shape(inputs)[1], 1), _single_seq_fn, _multi_seq_fn)


def crf_multitag_sequence_score(
    inputs: Tensors,
    tag_bitmap: Tensors,
    sequence_lengths: Tensors,
    transition_params: Tensors,
) -> tf.Tensor:
    """Computes the unnormalized score of all tag sequences matching
    tag_bitmap.
    tag_bitmap enables more than one tag to be considered correct at each time
    step. This is useful when an observed output at a given time step is
    consistent with more than one tag, and thus the log likelihood of that
    observation must take into account all possible consistent tags.
    Using one-hot vectors in tag_bitmap gives results identical to
    crf_sequence_score.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
          representing all active tags at each index for which to calculate the
          unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    tag_bitmap = K.cast(tag_bitmap, dtype='bool')
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')
    filtered_inputs = crf_filtered_inputs(inputs, tag_bitmap)

    # If max_seq_len is 1, we skip the score calculation and simply gather the
    # unary potentials of all active tags.
    def _single_seq_fn():
        return tf.reduce_logsumexp(filtered_inputs, axis=[1, 2], keepdims=False)

    def _multi_seq_fn():
        # Compute the logsumexp of all scores of sequences
        # matching the given tags.
        return crf_log_norm(
            inputs=filtered_inputs,
            sequence_lengths=sequence_lengths,
            transition_params=transition_params,
        )

    return K.switch(K.equal(K.shape(inputs)[1], 1), _single_seq_fn, _multi_seq_fn)


def crf_log_norm(
    inputs: Tensors, sequence_lengths: Tensors, transition_params: Tensors
) -> tf.Tensor:
    """Computes the normalization for a CRF.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input = inputs[:, :1, :]
    # first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = K.squeeze(first_input, axis=1)

    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp
    # over the "initial state" (the unary potentials).
    def _single_seq_fn():
        log_norm = tf.reduce_logsumexp(first_input, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm), log_norm
        )
        return log_norm

    def _multi_seq_fn():
        """Forward computation of alpha values."""
        rest_of_input = inputs[:, 1:, :]
        # rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
        # Compute the alpha values in the forward algorithm in order to get the
        # partition function.

        alphas = crf_forward(
            rest_of_input, first_input, transition_params, sequence_lengths
        )
        log_norm = tf.reduce_logsumexp(alphas, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm), log_norm
        )
        return log_norm

    return K.switch(K.equal(K.shape(inputs)[1], 1), _single_seq_fn, _multi_seq_fn)


def crf_log_likelihood(
    inputs: Tensors,
    tag_indices: Tensors,
    sequence_lengths: Tensors,
    transition_params: Optional[Tensors] = None,
) -> tf.Tensor:
    """Computes the log-likelihood of tag sequences in a CRF.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the log-likelihood.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix,
          if available.
    Returns:
      log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
        each example, given the sequence of tag indices.
      transition_params: A [num_tags, num_tags] transition matrix. This is
          either provided by the caller or created in this function.
    """
    # inputs = tf.convert_to_tensor(inputs)
    # cast type to handle different types
    tag_indices = K.cast(tag_indices, dtype='int32')
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')

    transition_params = K.cast(transition_params, K.dtype(inputs))
    sequence_scores = crf_sequence_score(
        inputs, tag_indices, sequence_lengths, transition_params
    )
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # Normalize the scores to get the log-likelihood per example.
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def crf_unary_score(
    tag_indices: Tensors, sequence_lengths: Tensors, inputs: Tensors
) -> tf.Tensor:
    """Computes the unary scores of tag sequences.
    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
    Returns:
      unary_scores: A [batch_size] vector of unary scores.
    """
    tag_indices = K.cast(tag_indices, dtype='int32')
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')

    batch_size = K.shape(inputs)[0]
    max_seq_len = K.shape(inputs)[1]
    num_tags = K.shape(inputs)[2]

    flattened_inputs = K.reshape(inputs, [-1])

    offsets = K.expand_dims(K.arange(0, batch_size) * max_seq_len * num_tags, 1)
    offsets += K.expand_dims(K.arange(0, max_seq_len) * num_tags, 0)

    flattened_tag_indices = K.reshape(offsets + tag_indices, [-1])

    unary_scores = K.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices), [batch_size, max_seq_len]
    )

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=K.shape(tag_indices)[1], dtype=unary_scores.dtype
    )

    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores


def crf_binary_score(
    tag_indices: Tensors, sequence_lengths: Tensors, transition_params: Tensors
) -> tf.Tensor:
    """Computes the binary scores of tag sequences.
    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      binary_scores: A [batch_size] vector of binary scores.
    """
    tag_indices = K.cast(tag_indices, dtype='int32')
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')

    num_tags = K.shape(transition_params)[0]
    num_transitions = K.shape(tag_indices)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    start_tag_indices = tag_indices[:, :num_transitions]
    # start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
    end_tag_indices = tag_indices[:, 1:num_transitions + 1]
    # end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    # Encode the indices in a flattened representation.
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = K.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = tf.gather(flattened_transition_params, flattened_transition_indices)

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=K.shape(tag_indices)[1], dtype=binary_scores.dtype
    )
    truncated_masks = masks[:, 1:]
    # truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)

    return binary_scores


def crf_forward(
    inputs: Tensors,
    state: Tensors,
    transition_params: Tensors,
    sequence_lengths: Tensors,
) -> tf.Tensor:
    """Computes the alpha values in a linear-chain CRF.
    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
         values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
    Returns:
      new_alphas: A [batch_size, num_tags] matrix containing the
          new alpha values.
    """
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')

    last_index = tf.maximum(
        tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1
    )
    inputs = tf.transpose(inputs, [1, 0, 2])
    transition_params = K.expand_dims(transition_params, 0)

    def _scan_fn(_state, _inputs):
        _state = K.expand_dims(_state, 2)
        transition_scores = _state + transition_params
        new_alphas = _inputs + tf.reduce_logsumexp(transition_scores, [1])
        return new_alphas

    all_alphas = tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
    # add first state for sequences of length 1
    all_alphas = K.concatenate([K.expand_dims(state, 1), all_alphas], 1)

    idxs = tf.stack([K.arange(0, K.shape(last_index)[0]), last_index], axis=1)
    return tf.gather_nd(all_alphas, idxs)


class AbstractRNNCell(keras.layers.Layer):
    """Abstract object representing an RNN cell.
    This is the base class for implementing RNN cells with custom behavior.
    Every `RNNCell` must have the properties below and implement `call` with
    the signature `(output, next_state) = call(input, state)`.
    Examples:
    ```python
        class MinimalRNNCell(AbstractRNNCell):
        def __init__(self, units, **kwargs):
            self.units = units
            super(MinimalRNNCell, self).__init__(**kwargs)
        @property
        def state_size(self):
            return self.units
        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer='uniform',
                                        name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')
            self.built = True
        def call(self, inputs, states):
            prev_output = states[0]
            h = K.dot(inputs, self.kernel)
            output = h + K.dot(prev_output, self.recurrent_kernel)
            return output, output
    ```
    This definition of cell differs from the definition used in the literature.
    In the literature, 'cell' refers to an object with a single scalar output.
    This definition refers to a horizontal array of such units.
    An RNN cell, in the most abstract setting, is anything that has
    a state and performs some operation that takes a matrix of inputs.
    This operation results in an output matrix with `self.output_size` columns.
    If `self.state_size` is an integer, this operation also results in a new
    state matrix with `self.state_size` columns.  If `self.state_size` is a
    (possibly nested tuple of) TensorShape object(s), then it should return a
    matching structure of Tensors having shape `[batch_size].concatenate(s)`
    for each `s` in `self.batch_size`.
    """

    def call(self, inputs, states):
        """The function that contains the logic for one RNN step calculation.
        Args:
        inputs: the input tensor, which is a slide from the overall RNN input by
            the time dimension (usually the second dimension).
        states: the state tensor from previous step, which has the same shape
            as `(batch, state_size)`. In the case of timestep 0, it will be the
            initial state user specified, or zero filled tensor otherwise.
        Returns:
        A tuple of two tensors:
            1. output tensor for the current timestep, with size `output_size`.
            2. state tensor for next step, which has the shape of `state_size`.
        """
        raise NotImplementedError('Abstract method')

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError('Abstract method')

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError('Abstract method')

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)


class CrfDecodeForwardRnnCell(AbstractRNNCell):
    """Computes the forward decoding in a linear-chain CRF."""

    @typechecked
    def __init__(self, transition_params: Tensors, **kwargs):
        """Initialize the CrfDecodeForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        super().__init__(**kwargs)
        self.supports_masking = True
        self._transition_params = K.expand_dims(transition_params, 0)
        #self._num_tags = K.shape(transition_params)[0]
        self._num_tags = K.int_shape(transition_params)[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def build(self, input_shape):
        super().build(input_shape)

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[List[Union[Tensors, None]], Tensors]:
        return mask

    def call(self, inputs: Tensors, state: Tensors, mask: Optional[Tensors] = None, **kwargs):
        """Build the CrfDecodeForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.
        Returns:
          backpointers: A [batch_size, num_tags] matrix of backpointers.
          new_state: A [batch_size, num_tags] matrix of new score values.
        """
        state = K.expand_dims(state[0], 2)
        transition_scores = state + K.cast(
            self._transition_params, K.dtype(state)
        )
        new_state = inputs + K.max(transition_scores, 1)
        backpointers = K.argmax(transition_scores, 1)
        backpointers = K.cast(backpointers, dtype='int32')
        return backpointers, new_state

    def get_config(self) -> dict:
        config = {
            "transition_params": K.squeeze(self._transition_params, axis=0).numpy().tolist()
        }
        base_config = super(CrfDecodeForwardRnnCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config: dict) -> "CrfDecodeForwardRnnCell":
        config["transition_params"] = np.array(
            config["transition_params"], dtype=np.float32
        )
        return cls(**config)


def crf_decode_forward(
    inputs: Tensors,
    state: Tensors,
    transition_params: Tensors,
    sequence_lengths: Tensors,
) -> tf.Tensor:
    """Computes forward decoding in a linear-chain CRF.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    sequence_lengths = K.cast(sequence_lengths, dtype='int32')
    # mask = tf.sequence_mask(sequence_lengths, K.shape(inputs)[1])
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params, dtype=K.dtype(inputs))
    '''
    # Use L.RNN
    crf_fwd_layer = keras.layers.RNN(
        crf_fwd_cell, return_sequences=True, return_state=True, stateful=False, dtype=K.dtype(inputs)
    )
    outputs, last_state = crf_fwd_layer(inputs, state)
    # Use L.RNN end
    '''
    # Use K.rnn
    (_, outputs, last_state) = K.rnn(crf_fwd_cell.call, inputs, [state])
    last_state = K.reshape(last_state, K.shape(state))
    # Use K.rnn end
    return outputs, last_state


def crf_decode_backward(inputs: Tensors, state: Tensors) -> tf.Tensor:
    """Computes backward decoding in a linear-chain CRF.
    Args:
      inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
      state: A [batch_size, 1] matrix of tag index of next step.
    Returns:
      new_tags: A [batch_size, num_tags]
        tensor containing the new tag indices.
    """
    inputs = tf.transpose(inputs, [1, 0, 2])

    def _scan_fn(state, inputs):
        state = K.squeeze(state, axis=1)
        idxs = tf.stack([K.arange(0, K.shape(inputs)[0]), state], axis=1)
        new_tags = K.expand_dims(tf.gather_nd(inputs, idxs), axis=-1)
        return new_tags

    return tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])


def crf_decode(
    potentials: Tensors, transition_params: Tensors, sequence_length: Tensors
) -> tf.Tensor:
    """Decode the highest scoring sequence of tags.
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of
                binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.
    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """
    sequence_length = K.cast(sequence_length, dtype='int32')

    # If max_seq_len is 1, we skip the algorithm and simply return the
    # argmax tag and the max activation.
    def _single_seq_fn():
        decode_tags = K.cast(K.argmax(potentials, axis=2), dtype='int32')
        best_score = K.reshape(tf.reduce_max(potentials, axis=2), shape=[-1])
        return decode_tags, best_score

    def _multi_seq_fn():
        # Computes forward decoding. Get last score and backpointers.
        initial_state = potentials[:, :1, :]
        # initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = K.squeeze(initial_state, axis=1)
        inputs = potentials[:, 1:, :]
        # inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

        sequence_length_less_one = tf.maximum(
            K.constant(0, dtype='int32'), sequence_length - 1
        )

        backpointers, last_score = crf_decode_forward(
            inputs, initial_state, transition_params, sequence_length_less_one
        )

        backpointers = tf.reverse_sequence(
            backpointers, sequence_length_less_one, seq_axis=1
        )

        initial_state = K.cast(K.argmax(last_score, axis=1), dtype='int32')
        initial_state = K.expand_dims(initial_state, axis=-1)

        decode_tags = crf_decode_backward(backpointers, initial_state)
        decode_tags = K.squeeze(decode_tags, axis=2)
        decode_tags = K.concatenate([initial_state, decode_tags], axis=1)
        decode_tags = tf.reverse_sequence(decode_tags, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)
        return decode_tags, best_score

    if K.int_shape(potentials)[1] is not None:
        # shape is statically know, so we just execute
        # the appropriate code path
        if K.int_shape(potentials)[1] == 1:
            return _single_seq_fn()
        else:
            return _multi_seq_fn()
    else:
        return K.switch(
            K.equal(K.shape(potentials)[1], 1), _single_seq_fn, _multi_seq_fn
        )


def crf_constrained_decode(
    potentials: Tensors,
    tag_bitmap: Tensors,
    transition_params: Tensors,
    sequence_length: Tensors,
) -> tf.Tensor:
    """Decode the highest scoring sequence of tags under constraints.
    This is a function for tensor.
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
          representing all active tags at each index for which to calculate the
          unnormalized score.
      transition_params: A [num_tags, num_tags] matrix of
                binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.
    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """

    filtered_potentials = crf_filtered_inputs(potentials, tag_bitmap)
    return crf_decode(filtered_potentials, transition_params, sequence_length)
