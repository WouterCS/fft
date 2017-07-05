# Import dataset functions
import RFNN.datasets.mnist as mnist
import RFNN.datasets.cifar10 as cifar10
from RFNN.datasets.utils import show_samples, split_dataset

def load_and_preprocess_dataset(nameDataset):
    if nameDataset == 'MNIST':
        return load_and_preprocess_MNIST()
    if nameDataset == 'cifar-10':
        return load_and_preprocess_CIFAR10()
    raise Exception('Unsupported dataset name')

def load_and_preprocess_MNIST():
    # Load dataset
    samples_data, samples_labels, test_data, test_labels = mnist.load_dataset()

    # Split in training and validation set
    train_data, train_labels, validation_data, validation_labels = split_dataset(samples_data, samples_labels, 60000)

    # Pre-process data: scale values between [0, 1]
    train_data[:, :, :, :] = train_data / 255.0
    validation_data[:, :, :, :] = validation_data / 255.0
    test_data[:, :, :, :] = test_data / 255.0

    # Get some properties of the dataset
    IMAGE_WIDTH = train_data.shape[2]
    IMAGE_HEIGHT = train_data.shape[1]
    IMAGE_DEPTH = train_data.shape[3]
    NUM_LABELS = 10

    dataset = {

        # Splitted sets
        'training_set': {'data': train_data, 'labels': train_labels},
        'validation_set': {'data': validation_data, 'labels': validation_labels},
        'test_set': {'data': test_data, 'labels': test_labels},

        # General info
        'width': IMAGE_WIDTH,
        'height': IMAGE_HEIGHT,
        'depth': IMAGE_DEPTH,
        'number_of_labels': NUM_LABELS,
    }

    return dataset
    
def load_and_preprocess_CIFAR10():
    return cifar10.load_and_preprocess_dataset()