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

import abc

from sklearn import svm
from sklearn import metrics


class Model(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, hparams, batch_size=None, num_classes=None,
                 summary_dir=None, verbose=False):
        """Initializes the model parameters.

        Args:
          hparams: The hyperparameters for the model as
            tf.contrib.training.HParams.
        """
        self._model = None
        self._hparams = hparams
        self._verbose = verbose
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._summary_dir = summary_dir

    @abc.abstractmethod
    def train(self, features, labels, seed=None):
        raise NotImplementedError('Not implemented')

    @abc.abstractmethod
    def evaluate(self, features, labels):
        raise NotImplementedError('Not implemented')
