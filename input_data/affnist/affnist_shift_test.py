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

"""Tests for affnist_shift."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import scipy.io as spio
import tensorflow as tf

import affnist_input_record
import affnist_shift


class AffnistShiftTest(tf.test.TestCase):

    def testReadImage(self):
        """Tests if the records are read in the expected order and value.

        Writes 3 images of size 40*40 into a temporary file. Calls the
        read_file function with the temporary file. Checks whether the order
        and value of pixels are correct for all 3 images.
        """
        colors = [0, 255, 100]
        height = 40
        size = height * height
        expecteds = [
            np.zeros((height, height)) + colors[0],
            np.zeros((height, height)) + colors[1],
            np.zeros((height, height)) + colors[2]
        ]
        image_filename = os.path.join(self.get_temp_dir(), 'affnist_image.mat')
        spio.savemat(image_filename, mdict={'image': expecteds})

        images = affnist_shift.read_file(image_filename)
        images = images['image'].reshape(len(colors), height, height)

        for i in range(len(colors)):
            self.assertAllEqual(expecteds[i], images[i])

    def testShift2d(self):
        """Tests if shifting of the image work as expected.

        Shifts the image in all direction with both positive and negative
        values.
        """
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_shifted_up_right_one = [[0, 4, 5], [0, 7, 8], [0, 0, 0]]
        expected_shifted_down_left_two = [[0, 0, 0], [0, 0, 0], [3, 0, 0]]
        shifted_one = affnist_shift.shift_2d(image, (-1, 1), 3)
        shifted_two = affnist_shift.shift_2d(image, (2, -2), 2)
        self.assertAllEqual(expected_shifted_up_right_one, shifted_one)
        self.assertAllEqual(expected_shifted_down_left_two, shifted_two)

    def testShift2dZero(self):
        """
        Tests if shifting of the image with max_shift 0 returns the image.
        """
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        shifted_zero = affnist_shift.shift_2d(image, (0, 0), 0)
        self.assertAllEqual(image, shifted_zero)


if __name__ == "__main__":
    tf.test.main()
