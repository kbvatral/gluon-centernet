import numpy as np
from mxnet import gluon

class Subdataset(gluon.data.Dataset):
    def __init__(self, dataset, size=0.25, random_seed=None, **kwargs):
        # To ignore numpy errors:
        #     pylint: disable=E1101
        super(Subdataset, self).__init__(**kwargs)
        
        self.dataset = dataset
        if size < 1:
            self.size = len(dataset) * size
        else:
            self.size = int(size)

        state = np.random.RandomState(random_seed)
        self.idxs = state.randint(0, len(dataset), self.size)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.idxs[idx])