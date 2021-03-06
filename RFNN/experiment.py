import tensorflow as tf
import numpy as np
from datetime import datetime

from RFNN.datasets.utils import show_samples, shuffle_samples, split_dataset
from custom_python_ops.custom_ops import tf_abs, tf_relu, tf_sqrt
from custom_python_ops.composite_ops import powMagnitude, sqrtMagnitude, tf_angle

from tensorflow.python.ops.spectral_ops import rfft2d, rfft
from tensorflow.python.ops.spectral_ops import irfft2d, irfft

from cifar_tf_model import cifar10_example_inference, train, tfCifarWeightsWeights

def error_rate(predictions, labels):
    # Return the error rate based on dense predictions and sparse labels
    error = 100.0 - (100.0 * np.mean(np.argmax(predictions, 1) == labels))

    return error


def eval_in_batches(data, sess, data_node, prediction_node, eval_batchsize, number_of_labels=10):
    """
    Utility function to evaluate a dataset by feeding batches of data to
    {eval_data} and pulling the results from {eval_predictions}.
    Saves memory and enables this to run on smaller GPUs.
    """
    
    # Get all predictions for a dataset by running it in small batches.
    size = data.shape[0]
    if size < eval_batchsize:
        print('Size: %d, eval_batchsize: %d' % (size, eval_batchsize))
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    print('size: %d, eval_batchsize: %d, type: %s' % (size, eval_batchsize, data.dtype), data.shape)
    predictions = np.ndarray(shape=(size, number_of_labels), dtype=np.float32)
    for begin in xrange(0, size, eval_batchsize):
        end = begin + eval_batchsize
        if end <= size:
            predictions[begin:end, :] = sess.run(prediction_node, feed_dict={data_node: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(prediction_node, feed_dict={data_node: data[-eval_batchsize:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]

    return predictions


def select_n_samples(data, labels, n, random_seed=None):
    """ Select randomly n samples from the set
    """

    # Shuffle first
    data, labels = shuffle_samples(data, labels, random_seed)

    # Return the first n samples
    return data[:n], labels[:n]


def create_basis_filters(grid, order, sigma, normalize, channels):

    basis = tf.user_ops.gaussian_basis_filters(grid=grid,
                                               order=order,
                                               sigma_x=sigma,
                                               sigma_y=sigma,
                                               normalize=normalize)
    basis = tf.expand_dims(basis, 2)
    basis = tf.tile(basis, [1, 1, channels, 1])

    return basis


def structured_conv_layer(images, basis, alphas):

    return tf.nn.separable_conv2d(images,
                                  depthwise_filter=basis,
                                  pointwise_filter=alphas,
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")

def evaluate(train_data, train_labels, validation_data, validation_labels, test_data, test_labels, sess, eval_data_node, prediction_eval, params, epoch):
     # Evaluate the model on different sets
    train_accuracy = None
    validation_accuracy = None
    test_accuracy = None

    if train_data.shape[0] != 0:
        train_accuracy = error_rate(
            eval_in_batches(train_data,
                            sess,
                            eval_data_node,
                            prediction_eval,
                            params.eval_batchsize),
            train_labels)

    if validation_data.shape[0] != 0:
        validation_accuracy = error_rate(
            eval_in_batches(validation_data,
                            sess,
                            eval_data_node,
                            prediction_eval,
                            params.eval_batchsize),
            validation_labels)

    if test_data.shape[0] != 0:
        test_accuracy = error_rate(
            eval_in_batches(test_data,
                            sess,
                            eval_data_node,
                            prediction_eval,
                            params.eval_batchsize),
            test_labels)

    # Append results
    params.acc_epochs.append(epoch)
    params.acc_train.append(train_accuracy)
    params.acc_val.append(validation_accuracy)
    params.acc_test.append(test_accuracy)
    print('Acc train: %f, acc test: %f' % (train_accuracy, test_accuracy))
    
def fftReLu(layerIn, params):
    if params.fftFunction == 'absFFT':
        layerIn = tf.transpose(layerIn, [0, 3, 1, 2])
        layerOut = irfft2d(tf.cast(tf.abs(rfft2d(layerIn)), tf.complex64))
        layerOut = tf.transpose(layerOut, [0, 2, 3, 1])
        return layerOut
    if params.fftFunction == 'absoluteValueUntransposed':
        return irfft2d(tf.cast(tf.abs(rfft2d(layerIn)), tf.complex64))
    if params.fftFunction == 'emptyFFT':
        return tf.nn.relu(irfft2d(rfft2d(layerIn)))
    if params.fftFunction == 'abs':
        return tf.abs(layerIn)
    if params.fftFunction == 'relu':
        return tf.nn.relu(layerIn)  
    if params.fftFunction == 'y-absFFT':
        layerIn = tf.transpose(layerIn, [0, 3, 1, 2])
        layerOut = irfft(tf.cast(tf.abs(rfft(layerIn)), tf.complex64))
        layerOut = tf.transpose(layerOut, [0, 2, 3, 1])
        return layerOut
    if params.fftFunction == 'x-absFFT':
        layerIn = tf.transpose(layerIn, [0, 3, 2, 1])
        layerOut = irfft(tf.cast(tf.abs(rfft(layerIn)), tf.complex64))
        layerOut = irfft(tf.cast(tf.abs(rfft(layerIn)), tf.complex64))
        layerOut = tf.transpose(layerOut, [0, 3, 2, 1])
        return layerOut
    if params.fftFunction == 'sqt-magnitude':
        layerIn = tf.transpose(layerIn, [0, 3, 2, 1])
        layerOut = irfft2d( sqrtMagnitude(rfft2d(layerIn) ))
        layerOut = tf.transpose(layerOut, [0, 2, 3, 1])
        return layerOut
    if params.fftFunction == 'powMagnitude':
        layerIn = tf.transpose(layerIn, [0, 3, 2, 1])
        layerOut = irfft2d( powMagnitude(rfft2d(layerIn), params))
        layerOut = tf.transpose(layerOut, [0, 2, 3, 1])
        return layerOut
    if params.fftFunction == 'identity':
        return layerIn

def printShape(shape):
    print('Dim: ', map(lambda x: x.value, shape))
    
def do_training(params, dataset):
    print('Do training: %s'  % str(datetime.now()))
    # Shorten some variable names
    train_data          = dataset['training_set']['data']
    train_labels        = dataset['training_set']['labels']
    validation_data     = dataset['validation_set']['data']
    validation_labels   = dataset['validation_set']['labels']
    test_data           = dataset['test_set']['data']
    test_labels         = dataset['test_set']['labels']

    # Modify training set
    train_data, train_labels = select_n_samples(train_data,
                                                train_labels,
                                                params.number_of_training_samples,
                                                params.seed)

    # Set the random seed
    tf.set_random_seed(params.seed)

    # Create placeholders
    train_data_node = tf.placeholder(tf.float32,
                                     shape=(params.batchsize,
                                            dataset['height'],
                                            dataset['width'],
                                            dataset['depth']),
                                     name="train_data_node")

    train_labels_node = tf.placeholder(tf.int64,
                                       shape=(params.batchsize,),
                                       name="train_labels_node")




    
    if params.model == 'model32to1':
        model = model32to1
        sizeFinalImage = 1*1
        weights = traditionalWeights(params, sizeFinalImage, dataset)
    elif params.model == 'model40to5':
        model = model40to5
        sizeFinalImage = 5*5
        weights = traditionalWeights(params, sizeFinalImage, dataset)
    elif params.model == 'cifar10_example_model':
        model = cifar10_example_model
        sizeFinalImage = 5*5
        weights = tfCifarWeightsWeights(params, sizeFinalImage, dataset)
    
    # Create all the trainable variables
    print('Create weights: %s'  % str(datetime.now()))

    
    # Define the loss function
    logits = model(params, train_data_node, weights, dataset['depth'], train=True)
    predition = tf.nn.softmax(logits)
    prediction_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits), name="loss")
    
    tf.add_to_collection('losses', prediction_loss)
    loss = tf.add_n(tf.get_collection('losses'))
    
    
    
    global_step = tf.Variable(0, trainable=False)
    if params.fixed_lr:
        learning_rate = params.initial_lr
    else:
        learning_rate = tf.train.exponential_decay(float(params.initial_lr), global_step, params.max_epochs *( params.number_of_training_samples // params.batchsize ), params.min_lr, staircase=False)
    print('Learning rate; starting value: %f, max epochs: %d, rate: %f' % (params.initial_lr, params.max_epochs, params.min_lr))

    temp_train_op = train(loss, global_step, params)
    
    # Create the optimizer
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
    # Create evaluation model
    eval_data_node = tf.placeholder(tf.float32,
                                    shape=(params.eval_batchsize,
                                           dataset['height'],
                                           dataset['width'],
                                           dataset['depth']))
    logits_eval = model(params, eval_data_node, weights, dataset['depth'], train=False)
    prediction_eval = tf.nn.softmax(logits_eval)

    # Create session
    sess = tf.Session()

    saver = tf.train.Saver()

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    print('Run session: %s'  % str(datetime.now()))

    batch_number = 0
    for curEpoch in range(int(np.ceil(params.max_epochs))):
        if hasattr(optimizer, '_lr_t'):
            cur_lr = optimizer._lr_t.eval(session=sess)
        else: # due to inconsistent naming in adagrad optimizer vs other optimizers
            if params.fixed_lr:
                cur_lr = optimizer._learning_rate
            else:
                cur_lr = optimizer._learning_rate.eval(session=sess)
        print('Epoch: %d, lr: %f, number of stepts: %d, at time: %s' % (curEpoch, cur_lr, global_step.eval(session=sess), str(datetime.now())))
        if curEpoch in params.eval_epochs:
            evaluate(train_data, train_labels, validation_data, validation_labels, test_data, test_labels, sess, eval_data_node, prediction_eval, params, curEpoch)
            params.learning_rate.append(cur_lr)
        if (params.save_freq is not None) and (curEpoch % params.save_freq == 0):
            params.save()
        # Shuffle the training samples between epochs
        train_data, train_labels = shuffle_samples(train_data, train_labels)

        number_of_batches_per_epoch = train_data.shape[0] // params.batchsize
        lossInEpoch = []
        for i in range(number_of_batches_per_epoch):
            
            # Compute epoch
            epoch = float(batch_number * params.batchsize) / train_data.shape[0]

            # Do evaluation

            # Select the training batch data and labels
            batch_data = train_data[i * params.batchsize:(i + 1) * params.batchsize]
            batch_labels = train_labels[i * params.batchsize:(i + 1) * params.batchsize]

            # Run 1 step of the gradient descent algorithm
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            #_, l = sess.run([train_op, loss], feed_dict=feed_dict)
            
            _, l = sess.run([temp_train_op, loss], feed_dict=feed_dict)
            
            # Increment batch number
            batch_number += 1

            # Update the results
            lossInEpoch.append(l)
        lossInEpoch = np.asarray(lossInEpoch)
        params.meanLoss.append(np.mean(lossInEpoch))
        params.medianLoss.append(np.median(lossInEpoch))
        params.varianceLoss.append(np.var(lossInEpoch))
        params.minLoss.append(np.min(lossInEpoch))
        params.maxLoss.append(np.max(lossInEpoch))
        print('Num batches: %d, min loss: %f, median loss: %f, mean loss: %f, variance loss: %f, max loss: %f' % (len(lossInEpoch), params.minLoss[-1], params.medianLoss[-1], params.meanLoss[-1], params.varianceLoss[-1], params.maxLoss[-1]))
    
    # Final evaluation on different sets
    evaluate(train_data, train_labels, validation_data, validation_labels, test_data, test_labels, sess, eval_data_node, prediction_eval, params, params.max_epochs)
    params.learning_rate.append(cur_lr)

    save_path = saver.save(sess, params.path_to_store_weights)

    
    # Create confusion matrix
    params.confusionMatrix = tf.confusion_matrix(labels = test_labels, 
                                                 predictions = np.argmax(
                                                        eval_in_batches(test_data,
                                                                        sess,
                                                                        eval_data_node,
                                                                        prediction_eval,
                                                                        params.eval_batchsize)
                                                 , 1),
                                                num_classes = 10).eval(session=sess)
    print(str(params.confusionMatrix))
    params
    # Close the session
    sess.close()

    
def model32to1(params, data, weights, inputDepth, train=False):

    if params.poolingLayer == 'max_pooling':
        poolFunction = tf.nn.max_pool
    if params.poolingLayer == 'avg_pooling':
        poolFunction = tf.nn.avg_pool

    # Dropout parameters
    KEEP_PROB_CONV  = params.KEEP_PROB_CONV
    KEEP_PROB_HIDDEN = params.KEEP_PROB_HIDDEN

    # Create basis filters
    basis1 = create_basis_filters(params.grid, params.order1, weights['s1'], params.normalize, inputDepth)
    basis2 = create_basis_filters(params.grid, params.order2, weights['s2'], params.normalize, params.N1)
    basis3 = create_basis_filters(params.grid, params.order3, weights['s3'], params.normalize, params.N2)
    basis4 = create_basis_filters(params.grid, params.order4, weights['s4'], params.normalize, params.N3)
    basis5 = create_basis_filters(params.grid, params.order5, weights['s4'], params.normalize, params.N4)

    # Block 0
    padSize = (32 - data.shape[1].value) / 2
    l0 = tf.pad(data, [[0, 0], [padSize, padSize], [padSize, padSize], [0, 0]], mode='CONSTANT')# Pad       32x32
    
    # Block 1
    l1 = structured_conv_layer(l0, basis1, weights['a1'])                                       # Conv
    #l1 = tf.nn.relu(l1)                                                                         # Relu
    l1 = fftReLu(l1, params)
    l1 = poolFunction(l1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      16x16
    l1 = tf.nn.local_response_normalization(l1, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l1 = tf.nn.dropout(l1, keep_prob=KEEP_PROB_CONV)                                  # Drop

    # Block 2
    l2 = structured_conv_layer(l1, basis2, weights['a2'])                                       # Conv
    #l2 = tf.nn.relu(l2)                                                                         # Relu
    l2 = fftReLu(l2, params)
    l2 = poolFunction(l2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      8x8
    l2 = tf.nn.local_response_normalization(l2, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l2 = tf.nn.dropout(l2, keep_prob=KEEP_PROB_CONV)                                  # Drop

    # Block 3
    l3 = structured_conv_layer(l2, basis3, weights['a3'])                                       # Conv
    #l3 = tf.nn.relu(l3)                                                                         # Relu
    l3 = fftReLu(l3, params)
    l3 = poolFunction(l3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      4x4
    l3 = tf.nn.local_response_normalization(l3, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l3 = tf.nn.dropout(l3, keep_prob=KEEP_PROB_CONV)                                  # Drop

    # Block 4
    l4 = structured_conv_layer(l3, basis4, weights['a4'])                                       # Conv
    #l4 = tf.nn.relu(l4)                                                                         # Relu
    l4 = fftReLu(l4, params)
    l4 = poolFunction(l4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      2x2
    l4 = tf.nn.local_response_normalization(l4, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l4 = tf.nn.dropout(l4, keep_prob=KEEP_PROB_CONV)                                  # Drop
    
    # Block 5
    l5 = structured_conv_layer(l4, basis5, weights['a5'])                                       # Conv
    #l5 = tf.nn.relu(l4)                                                                         # Relu
    l5 = fftReLu(l5, params)
    l5 = poolFunction(l5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      1x1
    l5 = tf.nn.local_response_normalization(l5, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l5 = tf.nn.dropout(l5, keep_prob=KEEP_PROB_CONV)                                  # Drop
    
    # Fully connected
    shape = l5.get_shape().as_list()
    l6 = tf.reshape(l5, [shape[0], shape[1] * shape[2] * shape[3]])                             # Flat      1x1
    if train: l6 = tf.nn.dropout(l6, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l6 = tf.matmul(l6, weights['fc_w1'])                                                        # FC
    l6 = l6 + weights['fc_b1']                                                                  # Bias

    return l6
 
def model40to5(params, data, weights, inputDepth, train=False):
    
    if params.poolingLayer == 'max_pooling':
        poolFunction = tf.nn.max_pool
    if params.poolingLayer == 'avg_pooling':
        poolFunction = tf.nn.avg_pool
        
    # Dropout parameters
    KEEP_PROB_CONV      = params.KEEP_PROB_CONV
    KEEP_PROB_HIDDEN    = params.KEEP_PROB_HIDDEN

    # Create basis filters
    basis1 = create_basis_filters(params.grid, params.order1, weights['s1'], params.normalize, inputDepth)
    basis2 = create_basis_filters(params.grid, params.order2, weights['s2'], params.normalize, params.N1)
    basis3 = create_basis_filters(params.grid, params.order3, weights['s3'], params.normalize, params.N2)

    # Block 0
    padSize = (40 - data.shape[1].value) / 2
    l0 = tf.pad(data, [[0, 0], [padSize, padSize], [padSize, padSize], [0, 0]], mode='CONSTANT')# Pad       40x40

    # Block 1
    l1 = structured_conv_layer(l0, basis1, weights['a1'])                                       # Conv
    #l1 = tf.nn.relu(l1)                                                                        # Relu
    l1 = fftReLu(l1, params)
    l1 = poolFunction(l1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      20x20
    l1 = tf.nn.local_response_normalization(l1, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l1 = tf.nn.dropout(l1, keep_prob=KEEP_PROB_CONV)                                  # Drop

    # Block 2
    l2 = structured_conv_layer(l1, basis2, weights['a2'])                                       # Conv
    #l2 = tf.nn.relu(l2)                                                                        # Relu
    l2 = fftReLu(l2, params)
    l2 = poolFunction(l2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      10x10
    l2 = tf.nn.local_response_normalization(l2, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l2 = tf.nn.dropout(l2, keep_prob=KEEP_PROB_CONV)                                  # Drop

    # Block 3
    l3 = structured_conv_layer(l2, basis3, weights['a3'])                                       # Conv
    #l3 = tf.nn.relu(l3)                                                                        # Relu
    l3 = fftReLu(l3, params)
    l3 = poolFunction(l3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")           # Pool      5x5
    l3 = tf.nn.local_response_normalization(l3, depth_radius=4, bias=2, alpha=1e-4, beta=0.75)  # Norm
    if train: l3 = tf.nn.dropout(l3, keep_prob=KEEP_PROB_CONV)                                  # Drop

    # Fully connected
    shape = l3.get_shape().as_list()
    l4 = tf.reshape(l3, [shape[0], shape[1] * shape[2] * shape[3]])                             # Flat      25x1
    if train: l4 = tf.nn.dropout(l4, keep_prob=KEEP_PROB_HIDDEN)                                # Drop
    l4 = tf.matmul(l4, weights['fc_w1'])                                                        # FC
    l4 = l4 + weights['fc_b1']                                                                  # Bias
    
    return l4
    
def cifar10_example_model(params, data, weights, inputDepth, train=False):
    return cifar10_example_inference(data, weights, params)
    
def traditionalWeights(params, sizeFinalImage, dataset):
    # Compute the number of basis filters
    F1 = (params.order1 + 1) * ((params.order1 + 1) - 1) / 2 + (params.order1 + 1)
    F2 = (params.order2 + 1) * ((params.order2 + 1) - 1) / 2 + (params.order2 + 1)
    F3 = (params.order3 + 1) * ((params.order3 + 1) - 1) / 2 + (params.order3 + 1)
    F4 = (params.order4 + 1) * ((params.order4 + 1) - 1) / 2 + (params.order4 + 1)
    F5 = (params.order5 + 1) * ((params.order5 + 1) - 1) / 2 + (params.order5 + 1)
    
    weights = {

        # Fully connected weights
        'fc_w1': tf.Variable(tf.random_normal([sizeFinalImage * params.N1, dataset['number_of_labels']],
                                              stddev=0.01,
                                              dtype=tf.float32)),
        'fc_b1': tf.Variable(tf.random_normal([dataset['number_of_labels']])),

        # Alphas
        'a1': tf.Variable(tf.random_uniform([1, 1, dataset['depth'] * F1, params.N1],
                                            minval=-1.0,
                                            maxval=1.0,
                                            dtype=tf.float32
                                            )),
        'a2': tf.Variable(tf.random_uniform([1, 1, params.N1 * F2, params.N2],
                                            minval=-1.0,
                                            maxval=1.0,
                                            dtype=tf.float32
                                            )),
        'a3': tf.Variable(tf.random_uniform([1, 1, params.N2 * F3, params.N3],
                                            minval=-1.0,
                                            maxval=1.0,
                                            dtype=tf.float32
                                            )),
        'a4': tf.Variable(tf.random_uniform([1, 1, params.N3 * F4, params.N4],
                                            minval=-1.0,
                                            maxval=1.0,
                                            dtype=tf.float32
                                            )),
        'a5': tf.Variable(tf.random_uniform([1, 1, params.N4 * F5, params.N5],
                                            minval=-1.0,
                                            maxval=1.0,
                                            dtype=tf.float32
                                            ))
    }

    if params.fixed_sigmas:
        weights['s1'] = tf.constant(params.initial_sigma1)
        weights['s2'] = tf.constant(params.initial_sigma2)
        weights['s3'] = tf.constant(params.initial_sigma3)
        weights['s4'] = tf.constant(params.initial_sigma4)
        weights['s5'] = tf.constant(params.initial_sigma5)
    else:
        weights['s1'] = tf.Variable(params.initial_sigma1)
        weights['s2'] = tf.Variable(params.initial_sigma2)
        weights['s3'] = tf.Variable(params.initial_sigma3)
    return weights