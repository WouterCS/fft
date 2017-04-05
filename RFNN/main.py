#imports for images
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from live_plotting import update_live_plots
#from functools import partial
import os.path
from __future__ import print_function
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
    
    trainGivenSetSize(dataset, 2000, 'adadelta', False, 10, 'absFFT', 1)

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
    
def trainGivenSetSize(dataset, numExamples, optimizer, fixed_lr, initial_lr, fftFunction, i):
    directory = '/results/results-%d-%d' % (numExamples, i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('Start of training with %d examples.' % (numExamples))
    params = para.parameters(directory + '/para')
    params.number_of_training_samples = numExamples
    params.optimizer = optimizer
    params.fixed_lr = fixed_lr
    params.initial_lr = initial_lr
    params.fftFunction = fftFunction
    
    print('Initialized the parameters: %s' % str(datetime.now()))
    do_training(params, dataset)
    
    with open(directory + '/README.txt', 'wb') as f:
        print('Training examples: %d' % numExamples, file = f)
        print('Epochs: %d' % params.max_epochs, file = f)
        print('Optimizer: %s' % optimizer, file = f)
        if fixed_lr:
            print('learning rate: %f' % params.initial_lr, file = f)
        else:
            print('learning rate: from %f to %f (exponential decay)' % (params.initial_lr, params.initial_lr * params.min_lr), file = f)
        print('non-linearity: %s' % fftFunction, file = f)
        
    
    params.save()
    #print('Final accuracy: %f' % params.acc_test[-1])