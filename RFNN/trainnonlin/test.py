import numpy as np
from RFNN.trainnonlin.generateData import loadData, generateData
import RFNN.trainnonlin.parameters as para
from RFNN.trainnonlin.training import do_training
from RFNN.dataset import load_and_preprocess_dataset
import tensorflow as tf

dataPath = '/data/storedData.npz'

def run():
    params = para.parameters('/home/wouter/Documents/git/fft/RFNN/trainnonlin/para')
    test_model(params)
    
    
def test_model(params):
    sess = tf.Session()
    
    new_saver = tf.train.import_meta_graph(params.saveDirectory + params.filename + '.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(params.saveDirectory))
    
    prediction = tf.get_default_graph().get_tensor_by_name("prediction:0")
    train_data_node = tf.get_default_graph().get_tensor_by_name("train_data_node:0")
    trainedWeights = sess.run(['fc_w1:0','fc_b1:0', 'fc_w2:0','fc_b2:0', 'fc_w3:0','fc_b3:0'])
    print('Maximum values in weights: %s' % str(map(lambda x: np.max(np.abs(x)), trainedWeights)))
    
    
    #testWithRandomInput(params, 100, sess, prediction, train_data_node)
    #testWithTrainingData(params, sess, prediction, train_data_node)
    testWithMNIST(params, sess, prediction, train_data_node)
    
def testWithTrainingData(params, sess, prediction, train_data_node):
    dataset = loadData(path)
    
    # Shorten some variable names
    train_data          = dataset['training_set']['data']
    train_labels        = dataset['training_set']['labels']

    # reshape to combine the MNIST-examples dimension and the kernal-dimension
    train_data = train_data.reshape((-1, 25, train_data.shape[2], train_data.shape[3], train_data.shape[4]))
    train_labels = train_labels.reshape((-1, 25, train_labels.shape[2], train_labels.shape[3], train_labels.shape[4]))
    
    checkLossForTestSet(params, train_data, train_labels, sess, prediction, train_data_node)

def testWithMNIST(params, sess, prediction, train_data_node):
    dataset = load_and_preprocess_dataset()
    test_data = dataset['test_set']['data']
    
    shape = test_data.shape
    test_data = np.transpose(test_data, (0, 3, 1, 2) ).reshape((-1,25, shape[1], shape[2], shape[3]))
    
    test_data = np.fft.rfft2(test_data).astype('complex64', casting = 'same_kind')
    test_labels = np.fft.rfft2(np.maximum(test_data, 0)).astype('complex64', casting = 'same_kind')
    
    checkLossForTestSet(params, test_data, test_labels, sess, prediction, train_data_node)

def testWithRandomInput(params, N, sess, prediction, train_data_node):
    randomImages = np.random.random((N, params.batchsize, 1, 28,28))
    testData = np.fft.rfft2(randomImages).astype('complex64', casting = 'same_kind')
    testLabels = np.fft.rfft2(np.maximum(randomImage, 0)).astype('complex64', casting = 'same_kind')
    
    checkLossForTestSet(params, testData, testLabels, sess, prediction, train_data_node)
    

def checkLossForTestSet(params, testData, testLabels, sess, prediction, train_data_node):
    # storedLoss = []
    # for i in range(len(testSet)):
        # randomImage = testSet[i]
        # input = np.fft.rfft2(randomImage).astype('complex64', casting = 'same_kind')
        # groundTruth = np.fft.rfft2(np.maximum(randomImage, 0)).astype('complex64', casting = 'same_kind')
        # pred = sess.run([prediction],feed_dict={train_data_node:input})[0]
        # loss = np.mean(np.absolute(pred - groundTruth), axis = (1,2,3))
        # storedLoss = np.concatenate((storedLoss,loss))
    # print('Max loss: %f, average loss: %f, median: %f' % (np.max(storedLoss), np.mean(storedLoss), np.median(storedLoss)))
    storedLoss = []
    for i in range(train_data.shape[0]):
        input = testData[i]
        pred = sess.run([prediction],feed_dict={train_data_node:input})[0]
        loss = np.mean(np.absolute(pred - testLabels[i]), axis = (1,2,3))
        storedLoss = np.concatenate((storedLoss,loss))
    print('Max loss: %f, average loss: %f, median: %f' % (np.max(storedLoss), np.mean(storedLoss), np.median(storedLoss)))