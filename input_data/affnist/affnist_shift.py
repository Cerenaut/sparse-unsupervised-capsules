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

"""Modules for preparing affNIST as input dataset.

It reads from affNIST raw MAT files, shifts and/or pads images, finally writes
the image and label pair as a tf.Example in a tfrecords file.

  Sample usage:
    python affnist_shift.py --data_dir=PATH_TO_AFFNIST_DIRECTORY
      --shift=2 --pad=0 --split=train
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import urllib
import zipfile

import numpy as np
import scipy.io as spio
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/affnist/input_data',
                       'Directory for storing input data')
tf.flags.DEFINE_integer('shift', 0, 'Maximum shift range.')
tf.flags.DEFINE_integer('pad', 0, 'Padding size.')
tf.flags.DEFINE_integer('max_shard', 0,
                        'Maximum number of examples in each file.')
tf.flags.DEFINE_string(
    'split', 'train',
    'The split of data to process: train, test, valid_train or valid_test.')
tf.flags.DEFINE_boolean('download', False, 'Download the dataset files.')

AFFNIST_FILES = {
    'train': 'training_and_validation_batches',
    'valid_train': 'training_batches',
    'valid_test': 'validation.mat',
    'test': 'test.mat'
}

AFFNIST_RANGE = {
    'train': (0, 1920000),  # 60,000 * 32 transformations
    'valid_train': (0, 1600000),  # 50,000 * 32 transformations
    'valid_test': (0, 320000),  # 10,000 * 32 transformations
    'test': (0, 320000)  # 10,000 * 32 transformations
}

AFFNIST_URL = 'http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/'

IMAGE_SIZE_PX = 40


def _download(data_dir, filename):
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        return

    print('Downloading {0}...'.format(filename))
    urllib.request.urlretrieve(AFFNIST_URL + filename, filepath)

    zip_ref = zipfile.ZipFile(filepath, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()


def int64_feature(value):
    """Casts value to a TensorFlow int64 feature list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """Casts value to a TensorFlow bytes feature list."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def shift_2d(image, shift, max_shift):
    """Shifts the image along each axis by introducing zero.

    Args:
      image: A 2D numpy array to be shifted.
      shift: A tuple indicating the shift along each axis.
      max_shift: The maximum possible shift.
    Returns:
      A 2D numpy array with the same shape of image.
    """
    max_shift += 1
    padded_image = np.pad(image, max_shift, 'constant')
    rolled_image = np.roll(padded_image, shift[0], axis=0)
    rolled_image = np.roll(rolled_image, shift[1], axis=1)
    shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
    return shifted_image


def shift_write_sharded_affnist(dataset, filename, shift, pad, max_shard):
    """Writes the transformed data as sharded tfrecords.

    Pads and shifts the data by adding zeros. Writes each pair of image and
    label as a tf.train.Example in a tfrecords file.

    Args:
      dataset: A list of tuples containing corresponding images and labels.
      filename: String, the name of the resultant tfrecord file.
      shift: Integer, the shift range for images.
      pad: Integer, the number of pixels to be padded
      max_shard: Integer, maximum number of examples in each shard.
    """
    images, labels = zip(*dataset)

    num_images = len(images)
    num_shards = int(np.ceil(num_images / FLAGS.max_shard))

    sharded_data = dict.fromkeys(range(num_shards))

    for i in range(num_shards):
        start = i * max_shard
        end = (i + 1) * max_shard

        sharded_data[i] = zip(images[start:end], labels[start:end])
        sharded_filename = (filename + '-{0}').format(i)

        shift_write_affnist(sharded_data[i], sharded_filename, shift, pad)


def shift_write_affnist(dataset, filename, shift, pad):
    """Writes the transformed data as tfrecords.

    Pads and shifts the data by adding zeros. Writes each pair of image and
    label as a tf.train.Example in a tfrecords file.

    Args:
      dataset: A list of tuples containing corresponding images and labels.
      filename: String, the name of the resultant tfrecord file.
      shift: Integer, the shift range for images.
      pad: Integer, the number of pixels to be padded
    """
    with tf.python_io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            padded_image = np.pad(image, pad, 'constant')
            for i in np.arange(-shift, shift + 1):
                for j in np.arange(-shift, shift + 1):
                    image_raw = shift_2d(
                        padded_image, (i, j), shift).tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': int64_feature(
                                                      IMAGE_SIZE_PX + 2 * pad),
                                'width': int64_feature(
                                                      IMAGE_SIZE_PX + 2 * pad),
                                'depth': int64_feature(1),
                                'label': int64_feature(label),
                                'image_raw': bytes_feature(image_raw),
                            }))
                    writer.write(example.SerializeToString())


def read_file(filepath):
    return spio.loadmat(filepath, struct_as_record=False, squeeze_me=True)


def read_mat_data(data_dir, split):
    """Extracts images and labels from affNIST MAT file.

    Reads the MAT file for the given split. Generates a
    tuple of numpy array containing the pairs of label and image.
    The format of the binary files are defined at:
      http://www.cs.toronto.edu/~tijmen/affNIST/

    Args:
      data_dir: String, the directory containing the dataset files.
      split: String, the dataset split to process. It can be one of train,
        test and valid.
    Returns:
      A list of (image, label). Image is a 40x40 numpy array and label is an
      int.
    """
    if split == 'train' or split == 'valid_train':
        mats = []
        data_dir = os.path.join(data_dir, AFFNIST_FILES[split])
        for file in os.listdir(data_dir):
            mats.append(
                read_file(os.path.join(data_dir, file)))

        images = np.stack([
            mats[i]['affNISTdata'].image.transpose() for i in range(len(mats))
        ])
        images = images.reshape(-1, *images.shape[-1:])

        labels = np.stack([
            mats[i]['affNISTdata'].label_int for i in range(len(mats))])
        labels = labels.reshape(-1)
    elif split == 'test' or split == 'valid_test':
        data = read_file(os.path.join(data_dir, AFFNIST_FILES[split]))
        images = data['affNISTdata'].image.transpose()
        labels = data['affNISTdata'].label_int

    start, end = AFFNIST_RANGE[split]
    images = images.reshape(end, IMAGE_SIZE_PX, IMAGE_SIZE_PX)

    return zip(images[start:], labels[start:])


def main(_):
    if FLAGS.download:
        _download(FLAGS.data_dir, AFFNIST_FILES[FLAGS.split] + '.zip')

    file_format = '{}_{}shifted_affnist.tfrecords'
    data = read_mat_data(FLAGS.data_dir, FLAGS.split)

    if FLAGS.max_shard > 0:
        file_format = 'sharded_' + file_format
    filename = os.path.join(FLAGS.data_dir,
                            file_format.format(FLAGS.split, FLAGS.shift))

    if FLAGS.max_shard > 0:
        shift_write_sharded_affnist(
            data, filename, FLAGS.shift, FLAGS.pad, FLAGS.max_shard)
    else:
        shift_write_affnist(data, filename, FLAGS.shift, FLAGS.pad)


if __name__ == '__main__':
    tf.app.run()
