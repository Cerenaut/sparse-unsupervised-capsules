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

"""Tests for affnist_input_record."""

import numpy as np
import tensorflow as tf

import affnist_input_record

AFFNIST_DATA_DIR = '../../testdata/affnist/'


class AffnistInputRecordTest(tf.test.TestCase):

    def testSingleTrain(self):
        with self.test_session(graph=tf.Graph()) as sess:
            features = affnist_input_record.inputs(
                data_dir=AFFNIST_DATA_DIR,
                batch_size=1,
                split='test',
                batch_capacity=2)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images, labels, recons_image = sess.run(
                [features['images'], features['labels'],
                 features['recons_image']])
            self.assertEqual((1, 10), labels.shape)
            self.assertEqual(1, np.sum(labels))
            self.assertItemsEqual([0, 1], np.unique(labels))
            self.assertEqual(40, features['height'])
            self.assertEqual((1, 40, 40, 1), images.shape)
            self.assertEqual(recons_image.shape, images.shape)
            self.assertAllEqual(recons_image, images)

            coord.request_stop()
            for thread in threads:
                thread.join()

    def testSingleTrainDistorted(self):
        with self.test_session(graph=tf.Graph()) as sess:
            features = affnist_input_record.inputs(
                data_dir=AFFNIST_DATA_DIR,
                batch_size=1,
                split='test',
                distort=True,
                batch_capacity=2)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images, labels, recons_image = sess.run(
                [features['images'], features['labels'],
                 features['recons_image']])
            self.assertEqual((1, 10), labels.shape)
            self.assertEqual(1, np.sum(labels))
            self.assertItemsEqual([0, 1], np.unique(labels))
            self.assertEqual(36, features['height'])
            self.assertEqual((1, 36, 36, 1), images.shape)
            self.assertEqual(recons_image.shape, images.shape)
            self.assertAllEqual(recons_image, images)

            coord.request_stop()
            for thread in threads:
                thread.join()

    def testSingleTestDistorted(self):
        with self.test_session(graph=tf.Graph()) as sess:
            features = affnist_input_record.inputs(
                data_dir=AFFNIST_DATA_DIR,
                batch_size=1,
                split='test',
                distort=True,
                batch_capacity=2,
                evaluate=True)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images, recons_image, recons_label = sess.run([
                features['images'], features['recons_image'],
                features['recons_label']
            ])
            self.assertEqual([0], recons_label)
            self.assertEqual(36, features['height'])
            self.assertEqual((1, 36, 36, 1), images.shape)
            self.assertAllEqual(recons_image, images)

            coord.request_stop()
            for thread in threads:
                thread.join()


if __name__ == '__main__':
    tf.test.main()
