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
    print('Start main: ', datetime.now().time())
    dataset = load_and_preprocess_dataset()
    print('Loaded the dataset: ', datetime.now().time())
    
    #trainGivenSetSize(dataset, 300,1)
    trainGivenSetSize(dataset, 1000,1)
    trainGivenSetSize(dataset, 2000, 1)
    trainGivenSetSize(dataset, 5000,1)
    #trainGivenSetSize(dataset, 10000)
    #trainGivenSetSize(dataset, 10000)

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
    
def trainGivenSetSize(dataset, numExamples, i):
    directory = '/results/results-%d-%d' % (numExamples, i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('Start of training with %d examples.' % (numExamples))
    params = para.parameters(directory + '/para')
    params.number_of_training_samples = numExamples
    print('Initialized the parameters: ', datetime.now().time())
    do_training(params, dataset)
    
    params.save()
    #print('Final accuracy: %f' % params.acc_test[-1])