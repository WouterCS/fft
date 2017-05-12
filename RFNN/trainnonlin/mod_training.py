#import tensorflow as tf
import numpy
from datetime import datetime
from RFNN.datasets.utils import select_n_samples
from RFNN.trainnonlin.generateData import loadData, generateData
import RFNN.trainnonlin.parameters as para
import tensorflow as tf

def test_do_training(path): # '/home/wouter/Documents/git/fft/RFNN/trainnonlin/storedData.npz')
    print('mod training')
    dataset = loadData(path)
    params = para.parameters('/home/wouter/Documents/git/fft/RFNN/trainnonlin/para')
    return do_training(params, dataset)

def do_training(params, dataset):
    print('Do training: %s'  % str(datetime.now()))
    
    # Shorten some variable names
    train_data          = dataset['training_set']['data']
    train_labels        = dataset['training_set']['labels']

    # reshape to combine the MNIST-examples dimension and the kernal-dimension
    train_data = train_data.reshape((train_data.shape[0] * train_data.shape[1], train_data.shape[2], train_data.shape[3], train_data.shape[4]))
    train_labels = train_labels.reshape((train_labels.shape[0] * train_labels.shape[1], train_labels.shape[2], train_labels.shape[3], train_labels.shape[4]))
    
    # Modify training set
    train_data, train_labels = select_n_samples(train_data,
                                                train_labels,
                                                params.number_of_training_samples,
                                                params.seed)
    # Set the random seed
    tf.set_random_seed(params.seed)
    
    # Create placeholders
    train_data_node = tf.placeholder(tf.complex64,
                                     shape=(params.batchsize,
                                            dataset['depth'],
                                            dataset['height'],
                                            dataset['width']),
                                     name="train_data_node")

    train_labels_node = tf.placeholder(tf.complex64,
                                       shape=(params.batchsize,
                                            dataset['depth'],
                                            dataset['height'],
                                            dataset['width']),
                                       name="train_labels_node")
     
    sizeImage = dataset['height'] * dataset['width'] * dataset['depth']
        
    weights = {

        # Fully connected weights, layer 1
        'fc_w1': tf.Variable(tf.complex( tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32),
                                                    tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32))),
        'fc_b1': tf.Variable(tf.complex(tf.random_normal([sizeImage]), tf.random_normal([sizeImage]))),
        }
    
    logits = model(params, train_data_node, weights, True)
    loss = tf.real(tf.norm(logits - train_labels_node))

    
    train_op = tf.train.AdamOptimizer(learning_rate=1.0).minimize(loss)
            
            
def model(params, data, weights, train=False):
    
    # Dropout parameters
    KEEP_PROB_HIDDEN = params.KEEP_PROB_HIDDEN
    shape = data.get_shape().as_list()
    
    l1 = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
    l1 = tf.matmul(l1, weights['fc_w1'])                                                        # FC
    l1 = l1 + weights['fc_b1']
    
    return tf.reshape(data, [shape[0], shape[1], shape[2], shape[3]])
    



