from __future__ import division
import os
from os.path import join as pjoin

import sys

import tensorflow as tf


IMAGE_PIXELS = 28 * 28
NUM_CLASSES = 10


flags = tf.app.flags
FLAGS = flags.FLAGS

def home_out(path):
  try:
      return pjoin(os.environ['HOME'], 'tmp', 'voy2', path)
  except KeyError:
      return pjoin(os.environ['HOMEPATH'], 'tmp', 'voy2', path)

# Autoencoder Architecture Specific Flags
flags.DEFINE_integer("num_hidden_layers", 3, "Number of hidden layers")

flags.DEFINE_integer('hidden1_units', 2000,
                     'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2_units', 2000,
                     'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3_units', 2000,
                     'Number of units in hidden layer 3.')

flags.DEFINE_integer('image_pixels', IMAGE_PIXELS, 'Total number of pixels')
flags.DEFINE_integer('num_classes', 10, 'Number of classes')

flags.DEFINE_float('pre_layer1_learning_rate', 0.001,
                   'Initial learning rate.')
flags.DEFINE_float('pre_layer2_learning_rate', 0.0005,
                   'Initial learning rate.')
flags.DEFINE_float('pre_layer3_learning_rate', 0.0002,
                   'Initial learning rate.')

flags.DEFINE_boolean('use_gaussian_noise', False,
                     'Whether to use gaussian noise instead of destructive noise.')

flags.DEFINE_float('noise_1', 0.05, 'Noise rate.')
flags.DEFINE_float('noise_2', 0.05, 'Noise rate.')
flags.DEFINE_float('noise_3', 0.05, 'Noise rate.')

# Data
flags.DEFINE_boolean('use_tf_contrib_learn_datasets', False,
                     'Whether to use tf.contrib.learn.datasets for learning data')

flags.DEFINE_string('filename_pattern', None,
                    'Read data from a filename glob pattern')

flags.DEFINE_integer('num_examples', None,
                     'Reduce the number of examples to this number')

# Performance
flags.DEFINE_boolean('no_finetuning', False,
                     'Skip fine-tuning (supervised) training step')

# Constants
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_integer('image_size', 28, 'Image square size')

flags.DEFINE_integer('batch_size', 100,
                     'Batch size. Must divide evenly into the dataset sizes.')

flags.DEFINE_float('supervised_learning_rate', 0.1,
                   'Supervised initial learning rate.')

flags.DEFINE_integer('pretraining_epochs', 60,
                     "Number of training epochs for pretraining layers")
flags.DEFINE_integer('finetuning_epochs', 56,
                     "Number of training epochs for "
                     "fine tuning supervised step")

flags.DEFINE_float('zero_bound', 1.0e-9,
                   'Value to use as buffer to avoid '
                   'numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                   'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')

# Directories
flags.DEFINE_string('log_dir_stem', 'tmp',
                    'Directory stem to put the tensor data.')

flags.DEFINE_string('data_dir', home_out('data'),
                    'Directory to put the training data.')

flags.DEFINE_string('summary_dir', home_out('summaries'),
                    'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', home_out('chkpts'),
                    'Directory to put the model checkpoints')

# TensorBoard
flags.DEFINE_boolean('no_browser', True,
                     'Whether to start browser for TensorBoard')

# Python
flags.DEFINE_string('python', sys.executable,
                    'Path to python executable')
