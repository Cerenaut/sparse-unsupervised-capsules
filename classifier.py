# Copyright (C) 2018 Project AGI
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import datetime
import itertools
import collections

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

from models.classifiers.svm_model import SvmModel
from models.classifiers.logistic_regression_model import (
    LogisticRegressionModel)


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', None, 'The data directory.')
tf.flags.DEFINE_string('summary_dir', None,
                       'Main directory for the experiments.')
tf.flags.DEFINE_string('model', 'logistic',
                       'Model to use for classification.')
tf.flags.DEFINE_string('dataset', 'mnist',
                       'The dataset to use for the experiment.'
                       'mnist, affnist or mnist-affnist.')
tf.flags.DEFINE_string('svm_hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'svm hparams of this experiment.')
tf.flags.DEFINE_string('logistic_hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'logistic hparams of this experiment.')

tf.flags.DEFINE_integer('seed', 42, 'Seed used for random shuffling.')
tf.flags.DEFINE_integer('last_step', 30000, 'The last step with a checkpoint.')

tf.flags.DEFINE_bool('verbose', True, 'Enable/disable verbose logging.')


def logistic_default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        C=100,
        eps=0.001,
        bias=True,
        penalty='l2',
        multi_class='ovr',
        solver='liblinear'
    )


def svm_default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        C=100,
        eps=0.001,
        gamma=0.1,
        kernel='rbf'
    )


def format_data(sess, features, labels):
    # Combine batch sizes
    features = features.reshape(-1, *features.shape[-2:])
    labels = labels.reshape(-1, *labels.shape[-1:])

    # Decode one-hot encoding
    labels = np.array([np.where(r == 1)[0][0] for r in labels])

    # Fatten features
    sess.run(tf.contrib.layers.flatten(features))

    return features, labels


def main(_):
    log_filename = '%s_%s_%s.log' % (
        datetime.datetime.now().strftime('%y%m%d-%H%M'),
        FLAGS.model, FLAGS.dataset)
    log_dir = os.path.join(FLAGS.summary_dir, 'results')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_filename)),
            logging.StreamHandler(sys.stdout)
        ])

    if FLAGS.dataset == 'affnist':
        train_filename = 'output_%s_test_%d.h5' % (
            FLAGS.dataset, FLAGS.last_step)
        test_filename = train_filename
    elif FLAGS.dataset == 'mnist-affnist':
        train_filename = 'output_%s_train_%d.h5' % (
            'mnist', FLAGS.last_step)
        test_filename = 'output_%s_test_%d.h5' % (
            'affnist', FLAGS.last_step)
    else:
        train_filename = 'output_%s_train_%d.h5' % (
            FLAGS.dataset, FLAGS.last_step)
        test_filename = 'output_%s_test_%d.h5' % (
              FLAGS.dataset, FLAGS.last_step)

    with h5py.File(os.path.join(FLAGS.data_dir, train_filename), 'r') as hf:
        train_features = hf['features'][:]
        train_labels = hf['labels'][:]

    with h5py.File(os.path.join(FLAGS.data_dir, test_filename), 'r') as hf:
        test_features = hf['features'][:]
        test_labels = hf['labels'][:]

    with tf.Session() as sess:
        X_train, y_train = format_data(sess, train_features, train_labels)
        X_test, y_test = format_data(sess, test_features, test_labels)

    if FLAGS.dataset == 'affnist':
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, train_size=60000, test_size=10000,
            random_state=FLAGS.seed, shuffle=False)
    elif FLAGS.dataset == 'mnist-affnist':
        X_test = X_test[60000:70000]
        y_test = y_test[60000:70000]

    # Flatten the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Dataset stats
    logging.info('X_train, y_train: {0}, {1}'.format(
        X_train.shape, y_train.shape))
    logging.info('X_test, y_test: {0}, {1}'.format(
        X_test.shape, y_test.shape))

    # Choose the model
    if FLAGS.model == 'svm':
        hparams = svm_default_hparams()
        if FLAGS.svm_hparams_override:
            hparams.parse(FLAGS.svm_hparams_override)

        model = SvmModel(hparams, verbose=FLAGS.verbose)
    elif FLAGS.model == 'logistic':
        hparams = logistic_default_hparams()
        if FLAGS.logistic_hparams_override:
            hparams.parse(FLAGS.logistic_hparams_override)

        model = LogisticRegressionModel(hparams, verbose=FLAGS.verbose)
    else:
        raise NotImplementedError(FLAGS.model + 'is not implemented.')

    # Training
    model.train(X_train, y_train, seed=FLAGS.seed)

    # Evaluation
    train_accuracy, train_preds = model.evaluate(X_train, y_train)
    test_accuracy, test_preds = model.evaluate(X_test, y_test)

    # Generate confusion matrices
    train_cm = metrics.confusion_matrix(y_train, train_preds)
    test_cm = metrics.confusion_matrix(y_test, test_preds)

    # Generate F1-score report
    train_f1 = metrics.classification_report(y_train, train_preds)
    test_f1 = metrics.classification_report(y_test, test_preds)

    # Print results
    logging.info('Training Accuracy: {0:f}%'.format(train_accuracy * 100))
    logging.info('Confusion Matrix')
    logging.info(train_cm)
    logging.info('F1 Score')
    logging.info(train_f1)

    logging.info('\n\n')

    logging.info('Test Accuracy: {0:f}%'.format(test_accuracy * 100))
    logging.info('Confusion Matrix')
    logging.info(test_cm)
    logging.info('F1 Score')
    logging.info(test_f1)


if __name__ == "__main__":
    tf.app.run()
