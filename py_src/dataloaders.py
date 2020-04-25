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


class mspec_net_speech_data(Dataset):

    def __init__(self, folder1, folder2, train=True):

        self.path1 = folder1
        self.path2 = folder2
        self.train = train

        self.files1 = listdir(self.path1)                               # NAM-Whisper
        self.files2 = listdir(self.path2)                               # Whisper-Speech

    def __getitem__(self, index):

        a = np.random.randint(len(self.files1))
        b = np.random.randint(len(self.files2))
       
        d1 = loadmat(join(self.path1, self.files1[int(a)]))             # NAM-Whisper
        d2 = loadmat(join(self.path2, self.files2[int(b)]))             # Whisper-Speech

        return np.array(d1['Feat']), np.array(d1['Clean_cent']),  np.array(d2['feat']), np.array(d2['Clean_cent'])

    def __len__(self):
        return len(self.files1)