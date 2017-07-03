from __future__ import print_function
import os.path
from datetime import datetime
import RFNN.parameters as para
from RFNN.experiment import do_training
from RFNN.dataset import load_and_preprocess_dataset
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train():
    print('Start main: %s'  % str(datetime.now()))
    dataset = load_and_preprocess_dataset()
    print('Loaded the dataset: %s'  % str(datetime.now()))
    
    
    class hyperParameters:
        def __init__(self):
            self.numExamples = 2000
            self.numEpochs = 600
            self.optimizer = 'adam' # 'adadelta'   'adam'   'adagrad'
            self.fixed_lr = True
            self.initial_lr = 1e-1
            self.fftFunction = 'absFFT'  # 'absFFT' 'absoluteValueUntransposed' 'emptyFFT'  'abs'   'relu'     'y-absFFT'     'x-absFFT'   'sqt-magnitude' 'custom_op'  'reference_op'   'powMagnitude' 'identity'
            self.model = 'model40to5'  #  'model40to5'    'model32to1'
            self.useDropout = True
            self.powMagnitude = 0.5
    
    hyperParam = hyperParameters()
    hyperParam.numEpochs = 200
    hyperParam.fixed_lr = False
    hyperParam.initial_lr = 1e-2
    hyperParam.numExamples = 5000
    hyperParam.optimizer = 'adam'
    hyperParam.fftFunction = 'powMagnitude'
    
    index = 0
    hyperParam.fftFunction = 'identity'
    index = index + 1
    trainGivenSetSize(dataset, hyperParam, index)
    
    hyperParam.fftFunction = 'relu'
    index = index + 1
    trainGivenSetSize(dataset, hyperParam, index)
    
    hyperParam.fftFunction = 'powMagnitude'
    hyperParam.powMagnitude = 0.9
    index = index + 1
    trainGivenSetSize(dataset, hyperParam, index)
    
    hyperParam.numExamples = 10000
    hyperParam.fftFunction = 'identity'
    index = index + 1
    trainGivenSetSize(dataset, hyperParam, index)
    
    hyperParam.fftFunction = 'relu'
    index = index + 1
    trainGivenSetSize(dataset, hyperParam, index)
    
    hyperParam.fftFunction = 'powMagnitude'
    hyperParam.powMagnitude = 0.9
    index = index + 1
    trainGivenSetSize(dataset, hyperParam, index)
    
    print('finished all training')

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
    if hyperParam.fftFunction == 'powMagnitude':
        print('Start of training with %d examples and %s non-linearity with power: %f.' % (hyperParam.numExamples, hyperParam.fftFunction, hyperParam.powMagnitude))
    else:
        print('Start of training with %d examples and %s non-linearity.' % (hyperParam.numExamples, hyperParam.fftFunction))
    params = para.parameters(directory + '/para', overwrite = True)
    params.number_of_training_samples = hyperParam.numExamples
    params.optimizer = hyperParam.optimizer
    params.fixed_lr = hyperParam.fixed_lr
    params.initial_lr = hyperParam.initial_lr
    params.fftFunction = hyperParam.fftFunction
    params.max_epochs = hyperParam.numEpochs
    params.model = hyperParam.model
    params.powMagnitude = hyperParam.powMagnitude
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
        if params.fftFunction == 'powMagnitude':
            print('non-linearity: %s, with power: %f' % (params.fftFunction, params.powMagnitude), file = f)
        else:
            print('non-linearity: %s' % params.fftFunction, file = f)
        
        print('Dropout conv layer: %f, dropout hidden layer: %f' % (params.KEEP_PROB_CONV, params.KEEP_PROB_HIDDEN), file = f)
    
    print('Initialized the parameters: %s' % str(datetime.now()))
    do_training(params, dataset)
    
    with open(directory + '/README.txt', 'a') as f:
        print('', file = f)
        print('Training finished', file = f)
        print('Final train error-rate: %f' % params.acc_train[-1], file = f)
        print('Final test error-rate: %f' % params.acc_test[-1], file = f)
        print('Confusion matrix:', file = f)
        print(str(params.confusionMatrix), file = f)
    
    plot_test_acc(params, directory)
    plot_loss(params, directory)
    
    print('Current run done.')
    
    params.save()
  

def plot_loss(params, directory):
    plt.clf()
    plt.plot(params.minLoss)
    plt.plot(params.medianLoss)
    plt.plot(params.maxLoss)
    plt.legend(['min loss in epoch', 'median loss in epoch', 'max loss in epoch'], loc='upper right')
    plt.figure(num =1, figsize = (20,20), dpi = 800)
    plt.xlabel('epochs')
    plt.ylabel('loss (%)')
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='black', linestyle='--')
    plt.yscale('log')
    #plt.ylim(0,100)
    plt.savefig(directory + '/loss.png')
    
def plot_test_acc(params, directory):
    plt.clf()
    plt.plot(params.acc_test, color = 'blue')
    plt.figure(num =1, figsize = (20,20), dpi = 800)
    plt.xlabel('epochs')
    plt.ylabel('test error-rate (%)')
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='black', linestyle='--')
    plt.ylim(0,100)
    plt.savefig(directory + '/conversionPlot.png')
    maxVal = 0
    if len(params.acc_test) > 50:
        plt.xlim(len(params.acc_test) - 50, len(params.acc_test))
    maxVal = np.max(params.acc_test[-50:])
    plt.ylim(0,min(15, maxVal * 1.2))
    plt.savefig(directory + '/conversionPlot-detailed.png')