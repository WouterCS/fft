#imports for images
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from live_plotting import update_live_plots
#from functools import partial
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
    
    trainGivenSetSize(dataset, 2000, 'adadelta', False, 1, 'absFFT', 1)
    trainGivenSetSize(dataset, 2000, 'adadelta', True, 3e-1, 'absFFT', 2)
    trainGivenSetSize(dataset, 2000, 'adadelta', True, 1e-1, 'absFFT', 3)
    trainGivenSetSize(dataset, 2000, 'adadelta', True, 3e-2, 'absFFT', 4)
    trainGivenSetSize(dataset, 2000, 'adadelta', True, 1e-2, 'absFFT', 5)
    trainGivenSetSize(dataset, 2000, 'adagrad', False, 1e-1, 'absFFT', 1)
    trainGivenSetSize(dataset, 2000, 'adagrad', True, 3e-2, 'absFFT', 6)
    trainGivenSetSize(dataset, 2000, 'adagrad', True, 1e-2, 'absFFT', 7)
    trainGivenSetSize(dataset, 2000, 'adagrad', True, 3e-3, 'absFFT', 8)

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
    
    params.save()
    #print('Final accuracy: %f' % params.acc_test[-1])