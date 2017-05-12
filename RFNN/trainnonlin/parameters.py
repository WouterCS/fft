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
        self.seed = 66478
        self.number_of_training_samples = 60000
        
        
        # training parameters
        self.batchsize = 2
        self.optimizer = 'adam'
        self.fixed_lr = True
        self.initial_lr = 1
        self.max_epochs = 10
        self.min_lr = 1e-1
        
        # model parameters
        self.weightType = 'complex'
        self.KEEP_PROB_HIDDEN = 0.3
        
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