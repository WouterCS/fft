import os.path
import pickle

class parameters:

    def __init__(self, filepath, overwrite = False):

        # Set the filepaths
        self.filepath = filepath

        path_without_extension = os.path.dirname(filepath) + "/" + os.path.splitext(os.path.basename(filepath))[0]
        self.path_model_normal = path_without_extension + "-normal.ckpt"
        self.path_model_resize = path_without_extension + "-resize.ckpt"

        # Load if file exists else set to default values
        if os.path.isfile(filepath) and not overwrite:
            self.load()
        else:
            self.reset()

    def reset(self):

        # Reset everything except filepaths

        #self.seed_normal = 66478
        #self.seed_resize = 66478
        self.seed = 66478
        #self.initial_sigma_normal = 1.0
        #self.initial_sigma_resize = 2.0

        # Model parameters
        self.grid = "discrete"
        self.normalize = True
        self.order1 = 3
        self.order2 = 2
        self.order3 = 2
        self.order4 = 2
        self.order5 = 2
        self.N1 = 64
        self.N2 = 64
        self.N3 = 64
        self.N4 = 64
        self.N5 = 64
        self.initial_sigma1 = 1.5
        self.initial_sigma2 = 1.0
        self.initial_sigma3 = 1.0
        self.initial_sigma4 = 1.0
        self.initial_sigma5 = 1.0
        self.fixed_sigmas = True
        self.fftFunction = 'absFFT'
        self.powMagnitude = 0.5
        self.model = 'model40to5'
        self.poolingLayer = 'max_pooling'
        self.KEEP_PROB_CONV = 0.8
        self.KEEP_PROB_HIDDEN = 0.3
        self.num_classes = 10

        # Training parameters
        self.max_epochs = 100
        self.batchsize = 25
        self.eval_epochs = range(2, 600) #[95,96,97,98,99,100]
        self.save_freq = 1
        self.eval_batchsize = 100
        self.number_of_training_samples = 100
        self.optimizer = 'adam'
        self.fixed_lr = False
        self.initial_lr = 1
        self.min_lr = 1.0e-1
        self.learning_rate = []

        # Regularization
        self.lambda_w1_normal = 0.0
        self.lambda_a1_normal = 0.0
        self.lambda_s1_normal = 0.0
        self.lambda_w1_resize = 0.0
        self.lambda_a1_resize = 0.0
        self.lambda_s1_resize = 0.0

        # Results
        self.path_to_store_weights = ''
        
        self.epochs = []
        self.meanLoss = []
        self.medianLoss = []
        self.varianceLoss = []
        self.minLoss = []
        self.maxLoss = []
        self.confusionMatrix = []

        self.acc_epochs = []
        self.acc_train = []
        self.acc_val = []
        self.acc_test = []

        print("Parameters reset to default values!")

    def load(self):
        restoredFilepath = self.filepath
        with open(self.filepath, 'rb') as f:
            self.__dict__ = pickle.load(f)
        self.filepath = restoredFilepath
        print("Parameters restored from: %s" % self.filepath)

    def save(self):
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.__dict__, f)
            print("Parameters saved to: %s" % self.filepath)
        except: # Exception,e:
            print('saving failed')#: %s' % str(e))