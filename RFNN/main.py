#imports for images
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from live_plotting import update_live_plots
#from functools import partial
from __future__ import print_function
import os.path

#print('line 8')
#import tensorflow as tf
#import numpy as np
from datetime import datetime
import RFNN.parameters as para
from RFNN.experiment import do_training
from RFNN.dataset import load_and_preprocess_dataset


def train():
    print('Start main: %s'  % str(datetime.now()))
    dataset = load_and_preprocess_dataset()
    print('Loaded the dataset: %s'  % str(datetime.now()))
    
    # ToDo vannacht:
    # trainGivenSetSize(dataset, 60000, 100, 'adadelta', False, 3, 'relu', 'model40to5', 7)
    
    
    numEpochs = 200
    trainGivenSetSize(dataset, 2000, numEpochs, 'adadelta', False, 3, 'x-absFFT', 'model40to5', 1)
    trainGivenSetSize(dataset, 2000, numEpochs, 'adadelta', False, 3, 'y-absFFT', 'model40to5', 2)
    trainGivenSetSize(dataset, 2000, numEpochs, 'adadelta', True, 3, 'x-absFFT', 'model40to5', 3)
    trainGivenSetSize(dataset, 2000, numEpochs, 'adadelta', True, 3, 'y-absFFT', 'model40to5', 4)
    trainGivenSetSize(dataset, 2000, numEpochs, 'adadelta', True, 0.3, 'x-absFFT', 'model40to5', 5)
    trainGivenSetSize(dataset, 2000, numEpochs, 'adadelta', True, 0.3, 'y-absFFT', 'model40to5', 6)
    trainGivenSetSize(dataset, 60000, 100, 'adadelta', False, 3, 'relu', 'model40to5', 7)
    # trainGivenSetSize(dataset, 1000, reluEpochs, 'adadelta', False, 3, 'relu', 2)
    # trainGivenSetSize(dataset, 2000, reluEpochs, 'adadelta', False, 3, 'relu', 3)
    # trainGivenSetSize(dataset, 5000, reluEpochs, 'adadelta', False, 3, 'relu', 4)
    # trainGivenSetSize(dataset, 10000, reluEpochs, 'adadelta', False, 3, 'relu', 5)
    #trainGivenSetSize(dataset, 20000, reluEpochs, 'adadelta', False, 3, 'relu', 6)
    #trainGivenSetSize(dataset, 60000, reluEpochs, 'adadelta', False, 3, 'relu', 7)
    
    # absFFTEpochs = 600
    # trainGivenSetSize(dataset, 2000,  absFFTEpochs, 'adadelta', True, 3, 'absFFT', 8)
    # trainGivenSetSize(dataset, 1000,  absFFTEpochs, 'adadelta', False, 3, 'absFFT', 9)
    # trainGivenSetSize(dataset, 2000,  absFFTEpochs, 'adadelta', False, 3, 'absFFT', 10)
    # trainGivenSetSize(dataset, 5000,  absFFTEpochs, 'adadelta', False, 3, 'absFFT', 11)
    # trainGivenSetSize(dataset, 10000,  absFFTEpochs, 'adadelta', False, 3, 'absFFT', 12)
    # trainGivenSetSize(dataset, 20000,  absFFTEpochs, 'adadelta', False, 3, 'absFFT', 13)
    # trainGivenSetSize(dataset, 60000,  absFFTEpochs, 'adadelta', False, 3, 'absFFT', 14)


def trainErrorRedo(dataset, numExamples):
    error = True
    i = 1
    while error:
        error = False
        try:
            trainGivenSetSize(dataset, numExamples, i)
        except Exception as e:
            error = True
            i = i + 1
    
def trainGivenSetSize(dataset, numExamples, numEpochs, optimizer, fixed_lr, initial_lr, fftFunction, model, i):
    directory = '/results/results-%d-%d' % (numExamples, i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('Start of training with %d examples and %s non-linearity.' % (numExamples, fftFunction))
    params = para.parameters(directory + '/para', overwrite = True)
    params.number_of_training_samples = numExamples
    params.optimizer = optimizer
    params.fixed_lr = fixed_lr
    params.initial_lr = initial_lr
    params.fftFunction = fftFunction
    params.max_epochs = numEpochs
    params.model = model
    
    with open(directory + '/README.txt', 'wb') as f:
        print('Training examples: %d' % numExamples, file = f)
        print('Epochs: %d' % params.max_epochs, file = f)
        print('Model: %s' % params.model, file = f)
        print('Optimizer: %s' % optimizer, file = f)
        if fixed_lr:
            print('learning rate: %f' % params.initial_lr, file = f)
        else:
            print('learning rate: from %f to %f (exponential decay)' % (params.initial_lr, params.initial_lr * params.min_lr), file = f)
        print('non-linearity: %s' % fftFunction, file = f)
    
    print('Initialized the parameters: %s' % str(datetime.now()))
    do_training(params, dataset)
    
    params.save()
    