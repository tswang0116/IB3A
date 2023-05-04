import warnings


class Default(object):
    pretrain_model_path = '../Datasets/imagenet-vgg-f.mat'
    batch_size = 128
    image_dim = 4096
    hidden_dim = 512
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 5
    max_epoch = 100

    # hyper-parameters
    alpha = 0.1  # from paper's Fig. 4
    beta = 1  # from paper's Fig. 4
    dataset_train_split = 0.5  # part of all data, that will be used for training
    dataset_query_split = 0.2  # part of evaluation data, that will be used for query
    use_aug_data = False


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