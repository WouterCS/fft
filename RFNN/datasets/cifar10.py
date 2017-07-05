import tensorflow as tf
import numpy as np
import sys 
import os

relativePathToData = './data/cifar-10'

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
        rawDict = unpickle(relativePathToData + '/data_batch_%d' % i)
        reshapeData = rawDict['data'].reshape((len(rawDict['labels']), 3, 32, 32))
        transposedData = reshapeData.transpose([0,2,3,1])
        trainData = np.concatenate((trainData,transposedData), axis = 0)
        trainLabels = np.concatenate((trainLabels, rawDict['labels']))
        
        
    rawTestData = unpickle(relativePathToData + '/test_batch')
    reshapeData = rawTestData['data'].reshape((len(rawTestData['labels']), 3, 32, 32))
    testData = reshapeData.transpose([0,2,3,1])
    testLabels = rawTestData['labels']

    metaInfo = unpickle(relativePathToData + '/batches.meta')

    dataset = {

        # Splitted sets
        'training_set': {'data': trainData, 'labels': trainLabels},
        'test_set': {'data': testData, 'labels': testLabels},

        # General info
        'width': trainData.shape[1],
        'height': trainData.shape[2],
        'depth': trainData.shape[3],
        'number_of_labels': trainData.shape[0],
        'label_names': metaInfo['label_names'],
    }
    
    return dataset