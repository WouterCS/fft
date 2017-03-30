import gzip
import os
import sys

import numpy
from six.moves import urllib

# User-defined constants
WORK_DIRECTORY = '/usr/local/lib/python2.7/dist-packages/RFNN/datasets/data/mnist'


def load_dataset():

    # Dataset specific properties
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    PIXEL_DEPTH = 255
    NUM_LABELS = 10

    def maybe_download(filename):
        # Download the data from Yann's website, unless it's already here.
        if not os.path.exists(WORK_DIRECTORY):
            os.mkdir(WORK_DIRECTORY)
        filepath = os.path.join(WORK_DIRECTORY, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath, reporthook)
            statinfo = os.stat(filepath)
            print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        return filepath

    def reporthook(blocknum, blocksize, totalsize):

        # reporthook from stackoverflow #13881092
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d / %d" % (
                percent, len(str(totalsize)), readsofar, totalsize)
            sys.stderr.write(s)
            if readsofar >= totalsize:  # near the end
                sys.stderr.write("\n")
        else:  # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))

    def extract_data(filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        No preprocessing: values are from [0, 255]
        """
        # print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
            data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
            data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            return data

    def extract_labels(filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        # print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        return labels

      # Get the data

    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    samples_data = extract_data(train_data_filename, 60000)
    samples_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    return samples_data, samples_labels, test_data, test_labels


if __name__ == "__main__":
    """
        Show example usage:
        load the dataset and show some images.
    """

    # Import utility functions
    import utils

    # Load the dataset
    samples_data, samples_labels, test_data, test_labels = load_dataset()

    # Show the first 100 samples
    utils.show_samples(samples_data[:100], samples_labels[:100])