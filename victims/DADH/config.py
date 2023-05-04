import warnings
import torch


class Default(object):
    pretrain_model_path = '../Datasets/imagenet-vgg-f.mat'

    # visualization
    image_dim = 4096
    hidden_dim = 8192
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 4

    # hyper-parameters
    alpha = 10
    gamma = 1
    beta = 1
    mu = 0.00005
    lamb = 1

    margin = 0.4
    dropout = False


    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)
            
        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                    print('\t{0}: {1}'.format(k, getattr(self, k)))




opt = Default()
