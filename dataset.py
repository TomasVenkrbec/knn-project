# Samples are saved in (sample_count, width, height, 3) shape
class Dataset:
    # https://www.cs.toronto.edu/~kriz/cifar.html
    # All files will be in dataset/CIFAR-100/, no subfolders
    # Train data files:
    #   train
    # Val data files:
    #   test
    def load_cifar100(self):
        # TODO: Load dataset
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.check_dataset()

    # ImageNet file structure
    # https://patrykchrabaszcz.github.io/Imagenet32/
    # All files will be in dataset/ImageNet/, no subfolders
    # Train data files:
    #   train_data_batch_1, train_data_batch_2, .., train_data_batch_10
    # Val data files:
    #   val_data
    def load_imagenet(self):
        # TODO: Load dataset
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.check_dataset()
    
    def check_dataset(self):
        # Train data have the same resolution and number of channels as validation data
        for dim_idx in range(1,4):
            assert self.x_train.shape[dim_idx] == self.x_val.shape[dim_idx]
        
        # Train data has same number of samples and labels
        assert self.x_train.shape[0] == self.y_train.shape[0]

        # Validation data has same number of samples and labels
        assert self.x_val.shape[0] == self.y_val.shape[0]

    @property
    def train_data(self):
        return (self.x_train, self.y_train)

    @property
    def val_data(self):
        return (self.x_val, self.y_val)

    @property
    def train_count(self):
        return self.x_train.shape[0]

    @property
    def val_count(self):
        return self.x_val.shape[0]

    @property
    def resolution(self):
        return self.x_train.shape[0]