# Sparse Unsupervised Capsules
The official source code for the SPARSECAPS model used in the following paper:
- "Sparse Unsupervised Capsules Generalize Better" by David Rawlinson, Abdelrahman Ahmed and Gideon Kowadlo.

## Requirements
- GPU
- [NumPy](http://www.numpy.org/)
- [TensorFlow](http://www.tensorflow.org) 1.5+
- See [REQUIREMENTS.txt](REQUIREMENTS.txt) for additional dependencies

## Quick Results
The checkpoint of the model trained on the expanded MNIST for affNIST generalizability
is [publicly available](https://storage.googleapis.com/project-agi/checkpoints/mnist_checkpoint.tar.gz)
to skip the training step and easily reproduce the experimental results.

## Usage

### Dataset Generation
Scripts to build necessary the data for training and/or evaluating the model
can be found in the `input_data` directory, for each dataset.

#### MNIST
To generate the shifted MNIST training set for training the model:
```
python mnist_shift.py --data_dir=PATH_TO_MNIST_DIRECTORY \
--split=train --shift=2 --pad=0
```

To generate the expanded MNIST training set for affNIST generalizability:
```
python mnist_shift.py --data_dir=PATH_TO_MNIST_DIRECTORY \
--split=train --shift=6 --pad=6
```

The test set can be generated in a similar way by using the following flags
instead: `--split=test --shift=0`. The dataset can also be downloaded from the
[source](http://yann.lecun.com/exdb/mnist/) by passing the `--download=true` flag.

#### affNIST
To generate the affNIST test set:
```
python affnist_shift.py --data_dir=PATH_TO_AFFNIST_DIRECTORY \
--split=test --shift=0 --pad=0
```

To generate the sharded affNIST test set to chunk the dataset over
separate `TFRecords` files:
```
python affnist_shift.py --data_dir=TFRecord \
--split=test --shift=0 --pad=0 --max_shard=80000
```

The `max_shard` is the maximum number of images in a single `TFRecords` file,
and since affNIST contains 320,000 images, this would generate 4 separate data files.
The dataset can also be downloaded from the [source](http://www.cs.toronto.edu/~tijmen/affNIST/)
by passing the `--download=true` flag.

### Model Workflow

#### Training

To train the model on the standard MNIST dataset:
```
python experiment.py --data_dir=/path/to/dataset/ \
--summary_dir=/path/to/log/dir --max_steps=30000 --dataset=mnist
--batch_size=128 --shift=2
```

To train on the expanded MNIST (40x40) for affNIST generalization:
```
python experiment.py --data_dir=/path/to/dataset/ \
--summary_dir=/path/to/log/dir --max_steps=30000 --dataset=mnist
--batch_size=128 --shift=6 --pad=6
```

Hyperparameters can be overriden using the `hparams_override` flag, e.g.
`--hparams_override=num_latent_capsules=24,num_atoms=16`. The flag should also be
used in the evaluation phase to ensure the model uses the expected parameters.

#### Encoding

To generate the encoded representation for a single dataset, e.g. MNIST:
```
python experiment.py --data_dir=/path/to/mnist_data/ --train=False \
--checkpoint=/path/to/model.ckpt --summary_dir=/path/to/output \
--eval_set=train --eval_size=60000
```

To generate the encoded representation for a sharded dataset, e.g. affNIST:
```
python experiment.py --data_dir=/path/to/mnist_data/ --train=False \
--checkpoint=/path/to/model.ckpt --summary_dir=/path/to/output \
--eval_set=test --eval_size=80000 --eval_shard=0
```

#### Classification
The classifier automatically finds the appropriate input data that was generated
by the encoder, so only the path to the encoded outputs is necessary.

Evaluate the encoded representation using SVM:
```
python classifier.py --data_dir=/path/to/outputs/dir \
--summary_dir=/path/to/log/dir --model=svm --dataset=mnist --last_step=30000
```

The SVM hyperparameters can also be overrided using a similar flag
`svm_hparams_override`.

## Acknowledgements
Thanks to Sabour et al. for open-sourcing the official [Capsules model](https://github.com/Sarasra/models/tree/master/research/capsules).
