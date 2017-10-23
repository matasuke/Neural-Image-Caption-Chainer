import numpy
import pickle
from chainer.dataset import dataset_mixin

class CaptionDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        self.train_data = dataset['train']
        self.val_data = dataset['val']
        self.test_data = dataset['test']

    def __len__(self):
        return len(self.train_data)

    def get_exapmle(self, i):
       pass 
