import numpy as np
from os import listdir
from os.path import join

from torch.utils.data import Dataset, DataLoader

from scipy.io import loadmat

# Class to load the parallel MCC features from .mat files into system
class parallel_dataloader(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(self.path)

        self.length = len(self.files)
        
    def __getitem__(self, index):
        d1 = loadmat(join(self.path, self.files[int(index)]))

        return  np.array(d1['Feat']), np.array(d1['Clean_cent'])
    
    def __len__(self):
        return self.length


# Class to load the non-parallel MCC features from .mat files into system
class non_parallel_dataloader(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(self.path)

        self.length = len(self.files)
        
    def __getitem__(self, index):
        d1 = loadmat(join(self.path, self.files[int(index)]))
        
        ind = np.random.randint(0, self.length)
        d2 = loadmat(join(self.path, self.files[int(ind)]))
        
        return  np.array(d1['Feat']), np.array(d2['Clean_cent'])
    
    def __len__(self):
        return self.length