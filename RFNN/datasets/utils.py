import numpy
import random


def shuffle_samples(data, labels, seed=None):
    """
    Shuffle the samples.

    :param data:
    :param labels:
    :return:
    """

    # Set the seed if required
    if seed is not None:
        random.seed(seed)

    n = data.shape[0]

    # Shuffle the index
    shuffled_index = range(n)
    random.shuffle(shuffled_index)

    # Shuffle the samples and the labels
    shuffled_data = data[shuffled_index]
    shuffled_labels = labels[shuffled_index]

    return shuffled_data, shuffled_labels


def split_dataset(data, labels, n1):
    """
    Split samples in train and validation set.

    :param data:
    :param labels:
    :param n1:
    :return:
    """

    n = data.shape[0]

    if n1 > n:
        raise Exception('The number of samples first set should be lower than the number of total samples')

    # Compute number of samples in set 2
    n2 = n - n1

    set1_data = data[0:n1, ...]
    set1_labels = labels[0:n1]
    set2_data = data[n1:n2, ...]
    set2_labels = labels[n1:n2]

    return set1_data, set1_labels, set2_data, set2_labels


def show_samples(fig, samples, labels=None):
    """
    Show samples in a grid, with in the top left corner the corresponding label

    :param samples:
    :param labels:
    :return:
    """

    # Squeeze gray scale images
    if samples.shape[3] == 1:
        samples = samples.squeeze()

    # Compute optimal grid size
    n = samples.shape[0]
    grid_size = int(numpy.ceil(numpy.sqrt(n)))

    from mpl_toolkits.axes_grid1 import ImageGrid
    grid = ImageGrid(fig, 111, nrows_ncols=(grid_size, grid_size), axes_pad=0)

    for i in range(grid_size * grid_size):
        if i < n:
            grid[i].imshow(samples[i], interpolation='nearest', cmap='gray')

            if labels is not None:
                grid[i].text(3,
                             3,
                             str(labels[i]),
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='red')

        grid[i].axis('off')

