from RFNN.dataset import load_and_preprocess_dataset
import numpy as np
from scipy.signal import convolve2d
import os

def splitComplexIntoReals(data):
        return data.view('Float64').reshape(data.shape+(2,)).transpose((5,0,1,2,3,4))

def convWithNRandom(data, n):
    
    randomFilter =  np.ndarray.tolist(np.random.random((n, 5, 5)))
    data = np.ndarray.tolist(np.transpose(data, (0, 3, 1, 2) ))
    
    #sampleImage = np.fft.rfft2(np.array(convolve2d( data[0][0], randomFilter[0] , 'same')))
    
    # output of size: (num MNIST examples, num random kernals, image depth, image height, image width)
    originalFilteredImage = np.empty((len(data), n, len(data[0]),  len(data[0][0]),  len(data[0][0][0])), dtype = 'float32')
    for i in range(0, len(data)):
        for j in range(0, n):
            for k in range(0, len(data[0])):
                # convolve   randomFilter[j, :, :] with data[i, k, :, :]
                originalFilteredImage[i,j,k,...] = np.array(convolve2d( data[i][k], randomFilter[j] , 'same'))
                
    print('Convolutions done')
    train_data = np.fft.rfft2(originalFilteredImage).astype('complex64', casting = 'same_kind')
    train_labels = np.fft.rfft2(np.maximum(originalFilteredImage, 0)).astype('complex64', casting = 'same_kind')
    return train_data, train_labels

def generateData(returnData = True, storeData = False, path = ''):
    dataset = load_and_preprocess_dataset()
    
    #split up the data
    train_data          = dataset['training_set']['data']
    train_labels        = dataset['training_set']['labels']
    validation_data     = dataset['validation_set']['data']
    validation_labels   = dataset['validation_set']['labels']
    test_data           = dataset['test_set']['data']
    test_labels         = dataset['test_set']['labels']
    
    (newtrain_data, newtrain_labels) = convWithNRandom(train_data[1:10,...], 1)
    
    datasetOut = {

        # Splitted sets
        'training_set': {'data': newtrain_data, 'labels': newtrain_labels},

        # General info
        'width': newtrain_data.shape[4],
        'height': newtrain_data.shape[3],
        'depth': newtrain_data.shape[2],
        'number_of_kernals': newtrain_data.shape[1],
        'number_of_MNIST': newtrain_data.shape[0]
    }
    
    if storeData:
        if not os.path.exists(directory):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            np.savez_compressed(path, data = datasetOut)
        loadData(path)
    else:
        return datasetOut
        
def loadData(path):
    with open(path, 'rb') as f:
        return np.load(f)['data'].tolist()
        
def generateDataFixedArg():
    return generateData(False, True, '/data/storedData.npz')