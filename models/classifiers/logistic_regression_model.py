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

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from models.classifiers import model


class LogisticRegressionModel(model.Model):

    def train(self, features, labels, seed=None):
        self._model = LogisticRegression(solver=self._hparams.solver,
                                         C=self._hparams.C,
                                         tol=self._hparams.eps,
                                         fit_intercept=self._hparams.bias,
                                         multi_class=self._hparams.multi_class,
                                         penalty=self._hparams.penalty,
                                         verbose=self._verbose,
                                         random_state=seed)

        self._model = self._model.fit(features, labels)

    def evaluate(self, features, labels):
        predictions = self._model.predict(features)
        accuracy = metrics.accuracy_score(labels, predictions)

        return accuracy, predictions
