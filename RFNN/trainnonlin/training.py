#import tensorflow as tf
import numpy
from datetime import datetime
from RFNN.datasets.utils import select_n_samples
from RFNN.trainnonlin.generateData import loadData, generateData
import RFNN.trainnonlin.parameters as para
import tensorflow as tf

def test_do_training(path): # '/home/wouter/Documents/git/fft/RFNN/trainnonlin/storedData.npz')
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
                                            dataset['height'],
                                            dataset['width'],
                                            dataset['depth']),
                                       name="train_labels_node")
     
    sizeImage = dataset['height'] * dataset['width'] * dataset['depth']
    if params.weightType == 'complex':
        weightType = tf.complex64
    elif params.weightType == 'real':
        weightType = tf.float32
        
    weights = {

        # Fully connected weights, layer 1
        'fc_w1': tf.Variable(tf.complex( tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32),
                                                    tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32))),
        'fc_b1': tf.Variable(tf.complex(tf.random_normal([sizeImage]), tf.random_normal([sizeImage]))),
        
        # Fully connected weights, layer 2
        'fc_w2': tf.Variable(tf.complex( tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32),
                                                    tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32))),
        'fc_b2': tf.Variable(tf.complex(tf.random_normal([sizeImage]), tf.random_normal([sizeImage]))),
        
        # Fully connected weights, layer 3
        'fc_w3': tf.Variable(tf.complex( tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32),
                                                    tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01
                                                        , dtype =  tf.float32))),
        'fc_b3': tf.Variable(tf.complex(tf.random_normal([sizeImage]), tf.random_normal([sizeImage])))
        }
    
    logits = model(params, train_data_node, weights, dataset['depth'], train=True)
    loss = tf.norm(logits - train_labels_node)
    
    global_step = tf.Variable(0, trainable=False)
    if params.fixed_lr:
        learning_rate = params.initial_lr
    else:
        learning_rate = tf.train.exponential_decay(float(params.initial_lr), global_step, params.max_epochs *( params.number_of_training_samples // params.batchsize ), params.min_lr, staircase=False)
        
    if params.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                               rho=0.95,
                                               epsilon=1e-06,
                                               name="optimizer")
    elif params.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif params.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    
    sess = tf.Session()
    
    for curEpoch in range(params.max_epochs):
        train_data, train_labels = shuffle_samples(train_data, train_labels)
        for i in range(int(np.floor(params.max_epochs / params.batchsize))):
            
            
            batch_data = train_data[i * params.batchsize:(i + 1) * params.batchsize]
            batch_labels = train_labels[i * params.batchsize:(i + 1) * params.batchsize]
            
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            _, l, w = sess.run([train_op, loss, weights], feed_dict=feed_dict)
            
            print('Loss: %f', l)
            
            
def model(params, data, weights, train=False):
    
    # Dropout parameters
    KEEP_PROB_CONV  = params.KEEP_PROB_CONV
    KEEP_PROB_HIDDEN = params.KEEP_PROB_HIDDEN
    shape = data.get_shape().as_list()
    
    l1 = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])                            
    if train: l1 = tf.nn.dropout(l1, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l1 = tf.matmul(l1, weights['fc_w1'])                                                        # FC
    l1 = l1 + weights['fc_b1']   
                            
    if train: l2 = tf.nn.dropout(l1, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l2 = tf.matmul(l2, weights['fc_w2'])                                                        # FC
    l2 = l2 + weights['fc_b2']   
                        
    if train: l3 = tf.nn.dropout(l2, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l3 = tf.matmul(l3, weights['fc_w3'])                                                        # FC
    l3 = l3 + weights['fc_b3']   
    l3 = tf.reshape([shape[0], shape[1], shape[2], shape[3]])
    
    return l3