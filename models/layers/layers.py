# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Modifications copyright (C) 2018 Project AGI
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

"""Library for capsule layers.

This has the layer implementation for coincidence detection, routing and
capsule layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from models.layers import variables


def _squash(input_tensor):
    """Applies norm nonlinearity (squash) to a capsule layer.

    Args:
      input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms]
        for a fully connected capsule layer or
        [batch, num_channels, num_atoms, height, width] for a convolutional
        capsule layer.

    Returns:
      A tensor with same shape as input (rank 3) for output of this layer.
    """
    with tf.name_scope('norm_non_linearity'):
        norm = tf.norm(input_tensor, axis=2, keepdims=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


def _leaky_routing(logits, output_dim):
    """Adds extra dimmension to routing logits.

    This enables active capsules to be routed to the extra dim if they are not
    a good fit for any of the capsules in layer above.

    Args:
      logits: The original logits. shape is
        [input_capsule_num, output_capsule_num] if fully connected. Otherwise,
        it has two more dimmensions.
      output_dim: The number of units in the second dimmension of logits.

    Returns:
      Routing probabilities for each pair of capsules. Same shape as logits.
    """

    # leak is a zero matrix with same shape as logits except dim(2) = 1 because
    # of the reduce_sum.
    leak = tf.zeros_like(logits, optimize=True)
    leak = tf.reduce_sum(leak, axis=2, keep_dims=True)
    leaky_logits = tf.concat([leak, logits], axis=2)
    leaky_routing = tf.nn.softmax(leaky_logits, dim=2)
    return tf.split(leaky_routing, [1, output_dim], 2)[1]


def _update_routing(votes, biases, logit_shape, num_dims, input_dim,
                    output_dim, num_routing, leaky):
    """Sums over scaled votes and applies squash to compute the activations.

    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.

    Args:
      votes: tensor, The transformed outputs of the layer below.
      biases: tensor, Bias variable.
      logit_shape: tensor, shape of the logit to be initialized.
      num_dims: scalar, number of dimmensions in votes. For fully connected
        capsule it is 4, for convolutional 6.
      input_dim: scalar, number of capsules in the input layer.
      output_dim: scalar, number of capsules in the output layer.
      num_routing: scalar, Number of routing iterations.
      leaky: boolean, if set use leaky routing.
      routing_factor: scalar, factor to decide additional routing iterations.
      unsupervised: boolean, whether or not the experiment is unsupervised

    Returns:
      The activation tensor of the output layer after either num_routing
      iterations, or (num_routing + (num_routing * routing_factor)) iterations
    """
    votes_t_shape = [3, 0, 1, 2]
    for i in range(num_dims - 4):
        votes_t_shape += [i + 4]
    r_t_shape = [1, 2, 3, 0]
    for i in range(num_dims - 4):
        r_t_shape += [i + 4]
    votes_trans = tf.transpose(votes, votes_t_shape)

    def _body(i, logits, activations, route):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        if leaky:
            route = _leaky_routing(logits, output_dim)
        else:
            route = tf.nn.softmax(logits, axis=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        # distances: [batch, input_dim, output_dim]
        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations, route)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    route = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    i, logits, activations, route = tf.while_loop(
        lambda i, logits, activations, route: i < num_routing,
        _body,
        loop_vars=[i, logits, activations, route],
        swap_memory=True)

    return activations.read(num_routing - 1), route


def capsule(input_tensor,
            input_dim,
            output_dim,
            layer_name,
            input_atoms=8,
            output_atoms=8,
            **routing_args):
    """Builds a fully connected capsule layer.

    Given an input tensor of shape `[batch, input_dim, input_atoms]`, this op
    performs the following:

      1. For each input capsule, multiples it with the weight variable to get
        votes of shape `[batch, input_dim, output_dim, output_atoms]`.
      2. Scales the votes for each output capsule by iterative routing.
      3. Squashes the output of each capsule to have norm less than one.

    Each capsule of this layer has one weight tensor for each capsules of layer
    below. Therefore, this layer has the following number of trainable
      variables:
        w: [input_dim * num_in_atoms, output_dim * num_out_atoms]
        b: [output_dim * num_out_atoms]

    Args:
      input_tensor: tensor, activation output of the layer below.
      input_dim: scalar, number of capsules in the layer below.
      output_dim: scalar, number of capsules in this layer.
      layer_name: string, Name of this layer.
      input_atoms: scalar, number of units in each capsule of input layer.
      output_atoms: scalar, number of units in each capsule of output layer.
      **routing_args: dictionary {leaky, num_routing}, args for routing
        function.

    Returns:
      Tensor of activations for this layer of shape
        `[batch, output_dim, output_atoms]`.
    """
    with tf.variable_scope(layer_name):
        # weights variable will hold the state of the weights for the layer
        weights = variables.weight_variable(
            [input_dim, input_atoms, output_dim * output_atoms])
        biases = variables.bias_variable([output_dim, output_atoms])
        with tf.name_scope('Wx_plus_b'):
            # Depthwise matmul: [b, d, c] ** [d, c, o_c] = [b, d, o_c]
            # To do this: tile input, do element-wise multiplication and reduce
            # sum over input_atoms dimmension.
            input_tiled = tf.tile(
                tf.expand_dims(input_tensor, -1),
                [1, 1, 1, output_dim * output_atoms])
            votes = tf.reduce_sum(input_tiled * weights, axis=2)
            votes_reshaped = tf.reshape(votes,
                                        [-1, input_dim, output_dim,
                                         output_atoms])
        with tf.name_scope('routing'):
            input_shape = tf.shape(input_tensor)
            logit_shape = tf.stack([input_shape[0], input_dim, output_dim])
            activations, route = _update_routing(
                votes=votes_reshaped,
                biases=biases,
                logit_shape=logit_shape,
                num_dims=4,
                input_dim=input_dim,
                output_dim=output_dim,
                **routing_args)
        return activations, route


def _depthwise_conv3d(input_tensor,
                      kernel,
                      input_dim,
                      output_dim,
                      input_atoms=8,
                      output_atoms=8,
                      stride=2,
                      padding='SAME'):
    """Performs 2D convolution given a 5D input tensor.

    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width]` squeezes the
    first two dimmensions to get a 4D tensor as the input of tf.nn.conv2d. Then
    splits the first dimmension and the last dimmension and returns the 6D
    convolution output.

    Args:
      input_tensor: tensor, of rank 5. Last two dimmensions representing height
        and width position grid.
      kernel: Tensor, convolutional kernel variables.
      input_dim: scalar, number of capsules in the layer below.
      output_dim: scalar, number of capsules in this layer.
      input_atoms: scalar, number of units in each capsule of input layer.
      output_atoms: scalar, number of units in each capsule of output layer.
      stride: scalar, stride of the convolutional kernel.
      padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels.

    Returns:
      6D Tensor output of a 2D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width]`,
        the convolution output shape and the input shape.
        If padding is 'SAME', out_height = in_height and out_width = in_width.
        Otherwise, height and width is adjusted with same rules as 'VALID' in
        tf.nn.conv2d.
    """
    with tf.name_scope('conv'):
        input_shape = tf.shape(input_tensor)
        _, _, _, in_height, in_width = input_tensor.get_shape()
        # Reshape input_tensor to 4D by merging first two dimmensions.
        # tf.nn.conv2d only accepts 4D tensors.

        input_tensor_reshaped = tf.reshape(input_tensor, [
            input_shape[0] *
            input_dim, input_atoms, input_shape[3], input_shape[4]
        ])
        input_tensor_reshaped.set_shape((None, input_atoms, in_height.value,
                                         in_width.value))
        conv = tf.nn.conv2d(
            input_tensor_reshaped,
            kernel,
            [1, 1, stride, stride],
            padding=padding,
            data_format='NCHW')
        conv_shape = tf.shape(conv)
        _, _, conv_height, conv_width = conv.get_shape()
        # Reshape back to 6D by splitting first dimmension to batch & input_dim
        # and splitting second dimmension to output_dim and output_atoms.

        conv_reshaped = tf.reshape(conv, [
            input_shape[0], input_dim, output_dim, output_atoms, conv_shape[2],
            conv_shape[3]
        ])
        conv_reshaped.set_shape((None, input_dim, output_dim, output_atoms,
                                 conv_height.value, conv_width.value))
        return conv_reshaped, conv_shape, input_shape


def conv_slim_capsule(input_tensor,
                      input_dim,
                      output_dim,
                      layer_name,
                      input_atoms=8,
                      output_atoms=8,
                      stride=2,
                      kernel_size=5,
                      padding='SAME',
                      **routing_args):
    """Builds a slim convolutional capsule layer.

    This layer performs 2D convolution given 5D input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width]`. Then refines
    the votes with routing and applies Squash non linearity for each capsule.

    Each capsule in this layer is a convolutional unit and shares its kernel
    over the position grid and different capsules of layer below. Therefore,
    number of trainable variables in this layer is:

      kernel: [kernel_size, kernel_size, input_atoms,
               output_dim * output_atoms]
      bias: [output_dim, output_atoms]

    Output of a conv2d layer is a single capsule with channel number of atoms.
    Therefore conv_slim_capsule is suitable to be added on top of a conv2d
    layer with num_routing=1, input_dim=1 and input_atoms=conv_channels.

    Args:
      input_tensor: tensor, of rank 5. Last two dimmensions representing height
        and width position grid.
      input_dim: scalar, number of capsules in the layer below.
      output_dim: scalar, number of capsules in this layer.
      layer_name: string, Name of this layer.
      input_atoms: scalar, number of units in each capsule of input layer.
      output_atoms: scalar, number of units in each capsule of output layer.
      stride: scalar, stride of the convolutional kernel.
      kernel_size: scalar, convolutional kernels are
        [kernel_size, kernel_size].
      padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels.
      **routing_args: dictionary {leaky, num_routing}, args to be passed to the
        update_routing function.

    Returns:
      Tensor of activations for this layer of shape
        `[batch, output_dim, output_atoms, out_height, out_width]`. If padding
        is 'SAME', out_height = in_height and out_width = in_width. Otherwise,
        height and width is adjusted with same rules as 'VALID' in
        tf.nn.conv2d.
    """
    with tf.variable_scope(layer_name):
        kernel = variables.weight_variable(shape=[
            kernel_size, kernel_size, input_atoms, output_dim * output_atoms
        ])
        biases = variables.bias_variable([output_dim, output_atoms, 1, 1])
        votes, votes_shape, input_shape = _depthwise_conv3d(
            input_tensor, kernel, input_dim, output_dim, input_atoms,
            output_atoms, stride, padding)

        with tf.name_scope('routing'):
            logit_shape = tf.stack([
                input_shape[0], input_dim, output_dim, votes_shape[2],
                votes_shape[3]
            ])
            biases_replicated = tf.tile(biases,
                                        [1, 1, votes_shape[2], votes_shape[3]])
            activations, route = _update_routing(
                votes=votes,
                biases=biases_replicated,
                logit_shape=logit_shape,
                num_dims=6,
                input_dim=input_dim,
                output_dim=output_dim,
                **routing_args)
        return activations, route


def _margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.
    Each wrong logit costs its distance to margin. For negative logits margin
    is 0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.
    Args:
      labels: tensor, one hot encoding of ground truth.
      raw_logits: tensor, model predictions in range [0, 1]
      margin: scalar, the margin after subtracting 0.5 from raw_logits.
      downweight: scalar, the factor for negative cost.
    Returns:
      A tensor with cost for each data point of shape [batch_size].
    """
    logits = raw_logits - 0.5
    positive_cost = labels * tf.cast(tf.less(logits, margin),
                                     tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def evaluate(logits, labels, num_targets, scope, loss_type, balance_factor,
             unsupervised):
    """Calculates total loss and performance metrics like accuracy.
    Args:
      logits: tensor, output of the model.
      labels: tensor, ground truth of the data.
      num_targets: scalar, number of present objects in the image,
        i.e. the number of 1s in labels.
      scope: The scope to collect losses of.
      loss_type: 'sigmoid' (num_targets > 1), 'softmax' or 'margin' for margin
        loss.
    Returns:
      The total loss of the model, number of correct predictions and number of
      cases where at least one of the classes is correctly predicted.
    Raises:
      NotImplementedError: if the loss_type is not softmax or margin loss.
    """
    total_loss = 0
    correct_sum = 0
    almost_correct_sum = 0

    if not unsupervised:
        with tf.name_scope('loss'):
            if loss_type == 'sigmoid':
                classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels / 2.0, logits=logits)
            elif loss_type == 'softmax':
                classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
            elif loss_type == 'margin':
                classification_loss = _margin_loss(
                    labels=labels, raw_logits=logits)
            else:
                raise NotImplementedError('Not implemented')

            with tf.name_scope('total'):
                batch_classification_loss = tf.reduce_mean(classification_loss)
                tf.add_to_collection('losses', batch_classification_loss)
        tf.summary.scalar('batch_classification_cost',
                          batch_classification_loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                _, targets = tf.nn.top_k(labels, k=num_targets)
                _, predictions = tf.nn.top_k(logits, k=num_targets)
                missed_targets = tf.contrib.metrics.set_difference(
                    targets, predictions)
                num_missed_targets = tf.contrib.metrics.set_size(
                    missed_targets)
                correct = tf.equal(num_missed_targets, 0)
                almost_correct = tf.less(num_missed_targets, num_targets)
                correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
                almost_correct_sum = tf.reduce_sum(
                    tf.cast(almost_correct, tf.float32))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('correct_prediction_batch', correct_sum)
        tf.summary.scalar('almost_correct_batch', almost_correct_sum)

    all_losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(all_losses, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    return total_loss, correct_sum, almost_correct_sum


def calculate_routing_coefficients(route, num_prime_capsules):
    """
    Computes the routing coefficients for each latent capsule.

    Args:
      route: tensor, the routing coefficients after N routing iterations.
      num_prime_capsules: scalar, number of primary capsules.

    Returns:
      An Tensor of shape [batch_size, num_latent_capsules] representing the
      routing coefficient of the latent capsule for every image in the batch.
    """
    dim = tf.sqrt(route.shape[1].value / num_prime_capsules)
    dim = tf.cast(dim, tf.int32)
    routing_coeffs = tf.reshape(
        route, [-1, dim, dim, num_prime_capsules, route.shape[2].value])
    capsule_routing = tf.reduce_max(routing_coeffs, axis=3)
    capsule_routing = tf.reduce_sum(capsule_routing, axis=2)
    return tf.reduce_sum(capsule_routing, axis=1)


def winning_frequency(ranks):
    """
    Computes the winning frequency for each latent capsule. The ranks are
    shaped [batch_size, num_latent_capsules]. The ranks are transposed into
    [num_latent_capsules, batch_size] where each row represents the rank
    received by each capsule for each image in the batch. It then simply counts
    the number of times the capsule won (i.e. rank == 0), and finally the
    frequency is computed by dividing over the batch size.

    This results in identifying how frequently a capsule is winning, in order
    to boost the losing capsules in later training.

    Args:
      ranks: The rank for each capsule based on routing coefficients, where
        0 is the highest and (num_latent_capsules - 1) is the lowest.

    Returns:
      An Tensor of shape [num_latent_capsules] where each element is the
      winning frequency of a capsule.
    """
    transposed_ranks = tf.transpose(ranks)  # num_latent_capsules, batch_size
    win_counts = tf.count_nonzero(tf.equal(transposed_ranks, 0),
                                  axis=1, dtype=tf.int32)
    return tf.cast(win_counts / tf.shape(transposed_ranks)[1], tf.float32)


def moving_average(shadow_variable, variable, decay):
    """
    Calculates the moving average using the classic formula employing the
    exponential decay. The shadow variable is initialised using the same value
    of the variable.

    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    Args:
      shadow_variable: The variable holding the moving averages.
      variable: The variable to calculate the average.
      decay: The exponential decay rate, typically close to 1.0.

    Returns:
      The expontential moving average of the variable.
    """
    return tf.cond(
        tf.equal(tf.reduce_sum(tf.abs(shadow_variable)), 0),
        lambda: variable,
        lambda: decay * shadow_variable + (1 - decay) * variable)


def boosting_op(num_latent_capsules, target_min_freq, target_max_freq,
                boost_factor, verbose):
    """
    The operation runs every N (boost_step) iterations to evalaute the current
    frequency moving average and boost/deboost the weights accordingly. The
    boosting allows lower performing capsules to get a chance to train.

    Args:
      num_latent_capsules: scalar, number of latent capsules.
      target_min_freq: scalar, minimum threshold for the average frequency.
      target_max_freq: scalar, maximum threshold for the average frequency.
      boost_factor: scalar, a factor to boost the weights by.
      verbose: boolean, visualises the variables in Tensorboard.

    Returns:
      The updated boosting weights.
    """
    freq_ema = variables.freq_ema_variable(num_latent_capsules)
    boosting_weights = variables.boosting_weights_variable(num_latent_capsules)

    def boost(c, boosting_weights, boost_factor):
        return tf.scatter_add(boosting_weights, c, boost_factor)

    def deboost(c, boosting_weights, boost_factor):
        boosting_weights = tf.scatter_sub(
            boosting_weights, c, boost_factor)
        return tf.scatter_update(
            boosting_weights, c, tf.maximum(1., boosting_weights[c]))

    for c in range(num_latent_capsules):
        boosting_weights = tf.cond(
            freq_ema[c] < target_min_freq,
            lambda: boost(c, boosting_weights, boost_factor),
            lambda: boost(c, boosting_weights, 0))
        boosting_weights = tf.cond(
            freq_ema[c] > target_max_freq,
            lambda: deboost(c, boosting_weights, boost_factor),
            lambda: deboost(c, boosting_weights, 0))

    return boosting_weights


def create_weighted_routing_mask(route, num_prime_capsules,
                                 num_latent_capsules, verbose, ema_decay=0.99,
                                 steepness_factor=12, clip_threshold=0.01):
    """
    Converts the capsule mask into an appropriate shape then applies it to
    the capsule embedding.

    Args:
      route: tensor, output of the last capsule layer.
      num_prime_capsules: scalar, number of primary capsules.
      num_latent_capsules: scalar, number of latent capsules.
      verbose: boolean, visualises the variables in Tensorboard.
      ema_decay: scalar, the expontential decay rate for the moving average.
      steepness_factor: scalar, controls the routing mask weights.
      clip_threshold: scalar, threshold for clipping values in the mask.

    Returns:
      The routing mask and the routing ranks.
    """
    capsule_routing = calculate_routing_coefficients(route, num_prime_capsules)

    freq_ema = variables.freq_ema_variable(
        num_latent_capsules, verbose)
    boosting_weights = variables.boosting_weights_variable(
        num_latent_capsules, verbose)

    # Boosting
    capsule_routing *= boosting_weights

    # Rank routing coefficients
    _, order = tf.nn.top_k(capsule_routing, num_latent_capsules)
    _, ranks = tf.nn.top_k(-order, num_latent_capsules)

    # Winning frequency
    freq = winning_frequency(ranks)
    freq_ema = tf.assign(freq_ema, moving_average(freq_ema, freq, ema_decay))
    ranks, _ = tf.tuple([ranks, freq_ema])

    # Normalise the rankings
    normalised_ranks = tf.cast(
        ranks / (num_latent_capsules - 1), dtype=tf.float32)

    routing_mask = tf.exp(-steepness_factor * normalised_ranks)

    # Set values < 0.1 to zero
    routing_mask = routing_mask - (
        routing_mask * tf.to_float(routing_mask < clip_threshold))

    return routing_mask, ranks


def _apply_capsule_mask(capsule_embedding, capsule_mask, num_atoms):
    """
    Converts the capsule mask into an appropriate shape then applies it to
    the capsule embedding.

    Args:
      capsule_embedding: tensor, output of the last capsule layer.
      capsule_mask: tensor, for each data in the batch it has the one hot
        encoding of the target id.
      num_atoms: scalar, number of atoms in the given capsule_embedding.

    Returns:
      The filtered capsule embedding after applying the mask.
    """
    capsule_mask_3d = tf.expand_dims(capsule_mask, -1)
    atom_mask = tf.tile(capsule_mask_3d, [1, 1, num_atoms])
    return capsule_embedding * atom_mask


def reconstruction(capsule_mask, num_atoms, capsule_embedding, layer_sizes,
                   num_pixels, reuse, image, balance_factor, unsupervised):
    """
    Adds the reconstruction loss and calculates the reconstructed image.

    Given the last capsule output layer as input of shape
    [batch, 10, num_atoms] add 3 fully connected layers on top of it.
    Feeds the masked output of the model to the reconstruction sub-network.
    Adds the difference with reconstruction image as reconstruction loss to the
    loss collection.

    Args:
      capsule_mask: tensor, for each data in the batch it has the one hot
        encoding of the target id.
      num_atoms: scalar, number of atoms in the given capsule_embedding.
      capsule_embedding: tensor, output of the last capsule layer.
      layer_sizes: (scalar, scalar), size of the first and second layer.
      num_pixels: scalar, number of pixels in the target image.
      reuse: if set reuse variables.
      image: The reconstruction target image.
      balance_factor: scalar, downweight the loss to be in valid range.
      unsupervised: boolean, reconstruction without true label & balance factor

    Returns:
      The reconstruction images of shape [batch_size, num_pixels].
    """
    first_layer_size, second_layer_size = layer_sizes

    if capsule_mask is not None:
        capsule_embedding = _apply_capsule_mask(
            capsule_embedding, capsule_mask, num_atoms)
    capsule_embedding_2d = tf.contrib.layers.flatten(capsule_embedding)

    reconstruction_2d = tf.contrib.layers.stack(
        inputs=capsule_embedding_2d,
        layer=tf.contrib.layers.fully_connected,
        stack_args=[(first_layer_size, tf.nn.relu),
                    (second_layer_size, tf.nn.relu),
                    (num_pixels, tf.sigmoid)],
        reuse=reuse,
        scope='recons',
        weights_initializer=tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32),
        biases_initializer=tf.constant_initializer(0.1))

    with tf.name_scope('loss'):
        image_2d = tf.contrib.layers.flatten(image)
        distance = tf.pow(reconstruction_2d - image_2d, 2)
        loss = tf.reduce_sum(distance, axis=-1)
        batch_loss = tf.reduce_mean(loss)
        if not unsupervised:
            balanced_loss = balance_factor * batch_loss
            tf.add_to_collection('losses', balanced_loss)
            tf.summary.scalar('balanced_reconstruction_error', balanced_loss)
        else:
            tf.add_to_collection('losses', batch_loss)
            tf.summary.scalar('reconstruction_error', batch_loss)

    return reconstruction_2d
