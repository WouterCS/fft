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
    
    #todo: model 32 doorrekenen
    class hyperParameters:
        def __init__(self):
            self.numExamples = 2000
            self.numEpochs = 600
            self.optimizer = 'adadelta' # 'adadelta'   'adam'   'adagrad'
            self.fixed_lr = True
            self.initial_lr = 1
            self.fftFunction = 'absFFT'  # 'absFFT'    'absoluteValueUntransposed'    'emptyFFT'    'abs'      'relu'     'y-absFFT'     'x-absFFT'
            self.model = 'model40to5'  #  'model40to5'    'model32to1'
            self.useDropout = True
    
    hyperParam = hyperParameters()

    
    hyperParam.numEpochs = 2
    hyperParam.fftFunction = 'relu'
    trainGivenSetSize(dataset, hyperParam, 1)



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
    
def trainGivenSetSize(dataset, hyperParam, i):
    directory = '/results/results-%d-%d' % (hyperParam.numExamples, i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('Start of training with %d examples and %s non-linearity.' % (hyperParam.numExamples, hyperParam.fftFunction))
    params = para.parameters(directory + '/para', overwrite = True)
    params.number_of_training_samples = hyperParam.numExamples
    params.optimizer = hyperParam.optimizer
    params.fixed_lr = hyperParam.fixed_lr
    params.initial_lr = hyperParam.initial_lr
    params.fftFunction = hyperParam.fftFunction
    params.max_epochs = hyperParam.numEpochs
    params.model = hyperParam.model
    if not hyperParam.useDropout:
        params.KEEP_PROB_CONV = 1.0
        params.KEEP_PROB_HIDDEN = 1.0
    
    with open(directory + '/README.txt', 'wb') as f:
        print('Training examples: %d' % params.number_of_training_samples, file = f)
        print('Epochs: %d' % params.max_epochs, file = f)
        print('Model: %s' % params.model, file = f)
        print('Optimizer: %s' % params.optimizer, file = f)
        if params.fixed_lr:
            print('learning rate: %f' % params.initial_lr, file = f)
        else:
            print('learning rate: from %f to %f (exponential decay)' % (params.initial_lr, params.initial_lr * params.min_lr), file = f)
        print('non-linearity: %s' % params.fftFunction, file = f)
        print('Dropout conv layer: %f, dropout hidden layer: %f' % (params.KEEP_PROB_CONV, params.KEEP_PROB_HIDDEN), file = f)
    
    print('Initialized the parameters: %s' % str(datetime.now()))
    do_training(params, dataset)
    
    with open(directory + '/README.txt', 'wb') as f:
        print('', file = f)
        print('Training finished', file = f)
        print('Final train error-rate: %f' % params.acc_train[-1], file = f)
        print('Final test error-rate: %f' % params.acc_test[-1], file = f)
        print('Confusion matrix:', file = f)
        print(str(params.confusionMatrix), file = f)
    
    print('Done with training.')
    
    params.save()
    