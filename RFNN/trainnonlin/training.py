#import tensorflow as tf
import numpy as np
from datetime import datetime
from RFNN.datasets.utils import select_n_samples, shuffle_samples
from RFNN.trainnonlin.generateData import loadData, generateData
import RFNN.trainnonlin.parameters as para
import tensorflow as tf

def test_do_training(path): # '/home/wouter/Documents/git/fft/RFNN/trainnonlin/storedData.npz')
    print('start')
    dataset = loadData(path)
    params = para.parameters('/home/wouter/Documents/git/fft/RFNN/trainnonlin/para')
    do_training(params, dataset)
    test_model(params)
    print('finish')
    
    
def test_model(params):
    
    sess = tf.Session()
    
    # The size of the image is twice as large, because we consider the real and imaginary parts seperately
    sizeImage = 2*1*28*15
    
    weights = defineWeights(sizeImage)
    
    new_saver = tf.train.import_meta_graph(params.saveDirectory + params.filename + '.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(params.saveDirectory))
    
    prediction = tf.get_default_graph().get_tensor_by_name("prediction:0")
    train_data_node = tf.get_default_graph().get_tensor_by_name("train_data_node:0")
    
    
    testWithRandomInput(weights, params, 100, sess, prediction, train_data_node)
    
def testWithRandomInput(weights, params, N, sess, prediction, train_data_node):
    
    randomImages = np.random.random((N, params.batchsize, 1, 28,28))
    checkLossForTestSet(weights, params, randomImages, sess, prediction, train_data_node)

def checkLossForTestSet(weights, params, testSet, sess, prediction, train_data_node):
    storedLoss = []
    for i in range(len(testSet)):
        randomImage = testSet[i]
        input = np.fft.rfft2(randomImage).astype('complex64', casting = 'same_kind')
        groundTruth = np.fft.rfft2(np.maximum(randomImage, 0)).astype('complex64', casting = 'same_kind')
        p = sess.run([prediction],feed_dict={train_data_node:input}).eval(session=sess)
        loss = tf.reduce_mean(tf.abs(p - groundTruth), axis = [2,3])#.eval(session=sess)
        storedLoss.append(loss)
    #print('Max loss: %f, average loss: %f' % (np.max(storedLoss), np.mean(storedLoss)))
    print(storedLoss)
    
def do_training(params, dataset):
    print('Do training: %s'  % str(datetime.now()))
    
    # Shorten some variable names
    train_data          = dataset['training_set']['data']
    train_labels        = dataset['training_set']['labels']

    # reshape to combine the MNIST-examples dimension and the kernal-dimension
    train_data = train_data.reshape((train_data.shape[0] * train_data.shape[1], train_data.shape[2], train_data.shape[3], train_data.shape[4]))
    train_labels = train_labels.reshape((train_labels.shape[0] * train_labels.shape[1], train_labels.shape[2], train_labels.shape[3], train_labels.shape[4]))
    print('Shape of training data: %s'  % str(train_data.shape))
    
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
    # The size of the image is twice as large, because we consider the real and imaginary parts seperately
    sizeImage = 2*dataset['height'] * dataset['width'] * dataset['depth']
    if params.weightType == 'complex':
        weightType = tf.complex64
    elif params.weightType == 'real':
        weightType = tf.float32
        
    weights = defineWeights(sizeImage)
        
    prediction = model(params, train_data_node, weights, train = True, tfData = True)
    prediction = tf.identity(prediction, name="prediction")
    loss = tf.reduce_mean(tf.abs(prediction - train_labels_node), axis = [1, 2,3])
    
    
    
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
    
    saver = tf.train.Saver(max_to_keep = 2)

    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for curEpoch in range(params.max_epochs):
        train_data, train_labels = shuffle_samples(train_data, train_labels)
        lossInCurEpoch = []
        print('Epoch: %d' % curEpoch)
        for i in range(int(np.floor(params.number_of_training_samples / params.batchsize))):
            
            batch_data = train_data[i * params.batchsize:(i + 1) * params.batchsize,...]
            batch_labels = train_labels[i * params.batchsize:(i + 1) * params.batchsize,...]

            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            _, l, w = sess.run([train_op, loss, weights], feed_dict=feed_dict)
            lossInCurEpoch = np.concatenate((lossInCurEpoch,l))
        saver.save(sess, params.saveDirectory + params.filename, global_step = global_step)
        sortedLoss = np.sort(lossInCurEpoch)
        print("Highest loss in epoch: %s" % str(sortedLoss[-10:]))
        print( "Indices of worst cases: %s" % str(np.argsort(lossInCurEpoch)[-10:]))
        print("Lowest loss in epoch: %s" % str(sortedLoss[:10]))
        print('In epoch %d, the average loss was: %f' % (curEpoch, np.mean(lossInCurEpoch)))
            
            
def model(params, data, weights, train=False, tfData = False):
    
    # Dropout parameters
    KEEP_PROB_HIDDEN = params.KEEP_PROB_HIDDEN
    if tfData:
        shape = data.get_shape().as_list()
    else:
        shape = list(data.shape)
        
    l0 = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
    l1 = tf.concat([tf.real(l0), tf.imag(l0)], axis = 1)
    if train: l1 = tf.nn.dropout(l1, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l1 = tf.matmul(l1, weights['fc_w1'])                                                        # FC
    l1 = l1 + weights['fc_b1']
    l2 = tf.nn.relu(l1)
                            
    if train: l2 = tf.nn.dropout(l2, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l2 = tf.matmul(l2, weights['fc_w2'])                                                        # FC
    l2 = l2 + weights['fc_b2']
    l3 = tf.nn.relu(l2)
                        
    if train: l3 = tf.nn.dropout(l3, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l3 = tf.matmul(l3, weights['fc_w3'])                                                        # FC
    l3 = l3 + weights['fc_b3']   

    treal, timag = tf.split(l3, 2, axis = 1)
    l4 = tf.complex(treal, timag)
    l4 = tf.reshape(l4, [shape[0], shape[1], shape[2], shape[3]])
    return l4
    
def dropoutForComplex(data, keep_prob):
    keep_prob = tf.convert_to_tensor(keep_prob,
                                                        dtype=tf.float32,
                                                        name="keep_prob")
    
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(tf.shape(data),
                                               dtype=tf.float32)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.floor(random_tensor)
    
    keep_prob = tf.cast(keep_prob, tf.complex64)
    binary_tensor = tf.cast(binary_tensor, tf.complex64)
    
    ret = tf.div(data, keep_prob) * binary_tensor
    ret.set_shape(data.get_shape())
    return ret

def defineWeights(sizeImage):
    weights = {

        # # Fully connected complex weights, layer 1
        # 'fc_w1': tf.Variable(tf.complex( tf.random_normal([sizeImage, sizeImage],
                                                        # stddev=0.01
                                                        # , dtype =  tf.float32),
                                                    # tf.random_normal([sizeImage, sizeImage],
                                                        # stddev=0.01
                                                        # , dtype =  tf.float32))),
        # 'fc_b1': tf.Variable(tf.complex(tf.random_normal([sizeImage]), tf.random_normal([sizeImage]))),
        
        # Fully connected complex weights, layer 1
        'fc_w1': tf.Variable(tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01, 
                                                        dtype =  tf.float32),
                                                        name = 'fc_w1'),
        'fc_b1': tf.Variable(tf.random_normal([sizeImage]), name = 'fc_b1'),
        
        # Fully connected complex weights, layer 1
        'fc_w2': tf.Variable(tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01, 
                                                        dtype =  tf.float32),
                                                        name = 'fc_w2'),
        'fc_b2': tf.Variable(tf.random_normal([sizeImage]), name = 'fc_b2'),
        
        # Fully connected complex weights, layer 1
        'fc_w3': tf.Variable(tf.random_normal([sizeImage, sizeImage],
                                                        stddev=0.01, 
                                                        dtype =  tf.float32),
                                                        name = 'fc_w3'),
        'fc_b3': tf.Variable(tf.random_normal([sizeImage]), name = 'fc_b3'),
        }
    return weights

