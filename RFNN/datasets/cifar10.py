import tensorflow as tf
import numpy as np
import sys 
import os

WORK_DIRECTORY = '/usr/local/lib/python2.7/dist-packages/RFNN/datasets/data/cifar-10'

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_and_preprocess_dataset():
    rawTrainData = []
    trainData = np.zeros((0,32,32,3), dtype = np.uint8)
    trainLabels = np.zeros((0), dtype = int)
    for i in range(1,6):
        rawDict = unpickle(WORK_DIRECTORY + '/data_batch_%d' % i)
        reshapeData = rawDict['data'].reshape((len(rawDict['labels']), 3, 32, 32))
        transposedData = reshapeData.transpose([0,2,3,1])
        trainData = np.concatenate((trainData,transposedData), axis = 0)
        trainLabels = np.concatenate((trainLabels, rawDict['labels']))
        
        
    rawTestData = unpickle(WORK_DIRECTORY + '/test_batch')
    reshapeData = rawTestData['data'].reshape((len(rawTestData['labels']), 3, 32, 32))
    testData = reshapeData.transpose([0,2,3,1])
    testLabels = rawTestData['labels']

    metaInfo = unpickle(WORK_DIRECTORY + '/batches.meta')

    dataset = {

        # Splitted sets
        'training_set': {'data': normalize(trainData), 'labels': trainLabels},
        'validation_set': {'data': np.empty((0, trainData.shape[1], trainData.shape[2], trainData.shape[3])), 'labels': np.empty((0))},
        'test_set': {'data': normalize(testData), 'labels': testLabels},

        # General info
        'width': trainData.shape[1],
        'height': trainData.shape[2],
        'depth': trainData.shape[3],
        'number_of_labels': len(metaInfo['label_names']),
        'label_names': metaInfo['label_names'],
    }
    
    return dataset
    
def normalize(data):
    data = data.astype(np.float64)
    return (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)