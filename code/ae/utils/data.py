"""Functions for downloading and reading MNIST data."""
from __future__ import division
from __future__ import print_function

import gzip

import os
import glob
import numpy
import tensorflow

from PIL import Image

from six.moves import urllib
from .flags import FLAGS

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

TEST_FRACTION = .2

def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>') # pylint disable=invalid-name
    return int(numpy.frombuffer(bytestream.read(4), dtype=dt))


def extract_images(filename):
    """Extract images from a gzip into a 4D uint8 numpy array [index, y, x, depth]."""
    print('\nExtracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        byterange = rows * cols * num_images
        buf = bytestream.read(byterange)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def read_images(filename_glob):
    """Read images from disk given a name pattern into a 4D uint8 numpy array [index, y, x, depth]."""
    i = 0
    image_filenames = glob.glob(filename_glob)
    with Image.open(image_filenames[i]) as firstimg:
        imgarr = numpy.array(firstimg)
        data = numpy.zeros([len(image_filenames), *(imgarr.shape), 1], dtype=numpy.uint8)
    for image_filename in image_filenames:
        with Image.open(image_filename) as img:
            imgarr = numpy.array(img)
            # Flatten to one band/channel.
            imgarr = imgarr.reshape(*(imgarr.shape),1)
            if data.shape[1:] != imgarr.shape:
                print('Image "{}" has the wrong shape. Expected: {}, Got: {}'
                      .format(image_filename, data.shape[1:], imgarr.shape))
            else:
                data[i] = imgarr
            i += 1
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    """ A DataSet, as in tensorflow.contrib.data """
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        """ The images array. """
        return self._images

    @property
    def labels(self):
        """ The labels array. """
        return self._labels

    @property
    def num_examples(self):
        """ The number of images and labels in this data set. """
        return self._num_examples

    @property
    def epochs_completed(self):
        """ The number of epochs fetched from this data set. """
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSetPreTraining(object):
    """ A data set, but for autoencoder pre-training (targets same shape as the input) """
    def __init__(self, images):
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._images[self._images < FLAGS.zero_bound] = FLAGS.zero_bound
        self._images[self._images > FLAGS.one_bound] = FLAGS.one_bound
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        """ The images array. """
        return self._images

    @property
    def num_examples(self):
        """ The number of images in this data set. """
        return self._num_examples

    @property
    def epochs_completed(self):
        """ The number of epochs fetched from this data set. """
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._images[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    # pylint: disable=invalid-name,attribute-defined-outside-init,protected-access,missing-docstring
    """ Reads data sets for supervised training. """
    class DataSets(object):
        pass
    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    
    if FLAGS.use_tf_contrib_learn_datasets:
        data_sets = tensorflow.contrib.learn.datasets.mnist.read_data_sets(train_dir=FLAGS.data_dir)
    else:
        if FLAGS.filename_pattern:
            all_images = read_images(FLAGS.filename_pattern)
            indices = numpy.random.permutation(all_images.shape[0])
            partition = int(all_images.shape[0] * (1-TEST_FRACTION))
            training_idx, test_idx = indices[:partition], indices[partition:]
            train_images = all_images[training_idx, :]
            test_images = all_images[test_idx, :]
            # These images are unlabeled.
            train_labels = numpy.zeros(train_images.shape[0])
            test_labels = numpy.zeros(train_images.shape[0])
            VALIDATION_SIZE = 50
        else:
            TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
            TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
            TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
            TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
            VALIDATION_SIZE = 5000

            local_file = maybe_download(TRAIN_IMAGES, train_dir)
            train_images = extract_images(local_file)

            local_file = maybe_download(TRAIN_LABELS, train_dir)
            train_labels = extract_labels(local_file, one_hot=one_hot)

            local_file = maybe_download(TEST_IMAGES, train_dir)
            test_images = extract_images(local_file)

            local_file = maybe_download(TEST_LABELS, train_dir)
            test_labels = extract_labels(local_file, one_hot=one_hot)

        validation_images = train_images[:VALIDATION_SIZE]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_images = train_images[VALIDATION_SIZE:]
        train_labels = train_labels[VALIDATION_SIZE:]

        data_sets.train = DataSet(train_images, train_labels)
        data_sets.validation = DataSet(validation_images, validation_labels)
        data_sets.test = DataSet(test_images, test_labels)

    if FLAGS.num_examples != None:
        data_sets.train._num_examples = FLAGS.num_examples
        data_sets.validation._num_examples = FLAGS.num_examples
        data_sets.test._num_examples = FLAGS.num_examples

    return data_sets


def read_data_sets_pretraining(train_dir):
    # pylint: disable=invalid-name,attribute-defined-outside-init,protected-access,missing-docstring
    """ Reads data sets for unsupervised pre-training. """
    if FLAGS.use_tf_contrib_learn_datasets:
        data_sets = tensorflow.contrib.learn.datasets.mnist.read_data_sets(train_dir=FLAGS.data_dir)
    else:
        class DataSets(object):
            pass
        data_sets = DataSets()

        if FLAGS.filename_pattern:
            all_images = read_images(FLAGS.filename_pattern)
            indices = numpy.random.permutation(all_images.shape[0])
            partition = int(all_images.shape[0] * (1-TEST_FRACTION))
            training_idx, test_idx = indices[:partition], indices[partition:]
            train_images = all_images[training_idx, :]
            test_images = all_images[test_idx, :]
            VALIDATION_SIZE = 50
        else:
            TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
            TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
            VALIDATION_SIZE = 5000

            local_file = maybe_download(TRAIN_IMAGES, train_dir)
            train_images = extract_images(local_file)

            local_file = maybe_download(TEST_IMAGES, train_dir)
            test_images = extract_images(local_file)

        validation_images = train_images[:VALIDATION_SIZE]
        train_images = train_images[VALIDATION_SIZE:]

        data_sets.train = DataSetPreTraining(train_images)
        data_sets.validation = DataSetPreTraining(validation_images)
        data_sets.test = DataSetPreTraining(test_images)

    if FLAGS.num_examples != None:
        data_sets.train._num_examples = FLAGS.num_examples
        data_sets.validation._num_examples = FLAGS.num_examples
        data_sets.test._num_examples = FLAGS.num_examples

    return data_sets


def _add_noise(x, rate):
    x_cp = numpy.copy(x)
    if FLAGS.use_gaussian_noise:
        x_cp += rate * numpy.random.normal(loc=0, scale=1, size=x_cp.shape)
        x_cp = numpy.maximum(numpy.minimum(x_cp,numpy.ones(x_cp.shape)),numpy.zeros(x_cp.shape))
        return x_cp
    else:
        pix_to_drop = numpy.random.random_sample(x_cp.shape) < rate
        x_cp[pix_to_drop] = FLAGS.zero_bound
        return x_cp


def fill_feed_dict_ae(data_set, input_pl, target_pl, noise=None):
    """Fills the feed_dict for pre-training the given step.
    A feed_dict takes the form of:
    feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
    }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        target_pl: The target placeholder, from placeholder_inputs().
        noise: The rate at which to add noise, or None.
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    input_feed, target_feed = data_set.next_batch(FLAGS.batch_size)
    if FLAGS.use_tf_contrib_learn_datasets:
        target_feed = input_feed
    if noise:
        input_feed = _add_noise(input_feed, noise)
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed
    }
    return feed_dict


def fill_feed_dict(data_set, images_pl, labels_pl, noise=False):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
    }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    if noise:
        images_feed = _add_noise(images_feed, FLAGS.drop_out_rate)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict
