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

"""Utility functions for declaring variables and adding summaries.

It adds all different scalars and histograms for each variable and provides
utility functions for weight and bias variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def weight_variable(shape, stddev=0.1, verbose=False):
    """Creates a CPU variable with normal initialization. Adds summaries.

    Args:
      shape: list, the shape of the variable.
      stddev: scalar, standard deviation for the initilizer.
      verbose: if set add histograms.

    Returns:
      Weight variable tensor of shape=shape.
    """
    with tf.device('/cpu:0'):
        with tf.name_scope('weights'):
            weights = tf.get_variable(
                'weights',
                shape,
                initializer=tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32),
                dtype=tf.float32)
            variable_summaries(weights, verbose)
    return weights


def bias_variable(shape, verbose=False):
    """Creates a CPU variable with constant initialization. Adds summaries.

    Args:
      shape: list, the shape of the variable.
      verbose: if set add histograms.

    Returns:
      Bias variable tensor with shape=shape.
    """
    with tf.device('/cpu:0'):
        with tf.name_scope('biases'):
            biases = tf.get_variable(
                'biases',
                shape,
                initializer=tf.constant_initializer(0.1),
                dtype=tf.float32)
            variable_summaries(biases, verbose)
    return biases


def freq_ema_variable(shape, verbose=False):
    """Creates a CPU variable with constant initialization. Adds summaries.

    Args:
      shape: list, the shape of the variable.
      verbose: if set add histograms.

    Returns:
      Frequency moving average variable tensor with shape=shape.
    """
    with tf.device('/cpu:0'):
        with tf.variable_scope('freq_ema', reuse=tf.AUTO_REUSE):
            freq_ema = tf.get_variable(
                'freq_ema',
                shape,
                initializer=tf.constant_initializer(0),
                dtype=tf.float32,
                trainable=False)
            capsule_summaries(freq_ema, shape, verbose)
    return freq_ema


def boosting_weights_variable(shape, verbose=False):
    """Creates a CPU variable with constant initialization. Adds summaries.

    Args:
      shape: list, the shape of the variable.
      verbose: if set add histograms.

    Returns:
      Boosting weights variable tensor with shape=shape.
    """
    with tf.device('/cpu:0'):
        with tf.variable_scope('boosting_weights', reuse=tf.AUTO_REUSE):
            boosting_weights = tf.get_variable(
                'boosting_weights',
                shape,
                initializer=tf.constant_initializer(1),
                dtype=tf.float32,
                trainable=False)
            capsule_summaries(boosting_weights, shape, verbose)
    return boosting_weights


def variable_summaries(var, verbose):
    """Attaches a lot of summaries to a Tensor (for TensorBoard visualization).

    Args:
      var: tensor, statistic summaries of this tensor is added.
      verbose: if set add histograms.
    """
    if verbose:
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    else:
        pass


def capsule_summaries(var, num_capsules, verbose, name='capsule'):
    """Attaches a lot of summaries to a Tensor (for TensorBoard visualization).

    Args:
      var: tensor, statistic summaries of this tensor is added.
      verbose: if set add scalar plots.
    """
    if verbose:
        for c in range(num_capsules):
            tf.summary.scalar(str(name) + '_' + str(c), var[c])
    else:
        pass


def activation_summary(x, verbose):
    """Creates summaries for activations.

    Creates a summary that provides a histogram and sparsity of activations.

    Args:
      x: Tensor
      verbose: if set add histograms.
    """
    if verbose:
        tf.summary.histogram('activations', x)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))
    else:
        pass
