import numpy as np
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import visdom
import math
import matplotlib.pyplot as plt 
import scipy
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

from dataloaders import parallel_dataloader
from networks import dnn
from utils import *

# Connect with Visdom for the loss visualization
viz = visdom.Visdom()

# Path where you want to store your results        
mainfolder = "../dataset/features/US_102/batches/VUV/"
checkpoint = "../results/checkpoints/vuv/"

# Training Data path
traindata = parallel_dataloader(folder_path=mainfolder)
train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)  # For windows keep num_workers = 0


# Path for validation data
valdata = parallel_dataloader(folder_path=mainfolder)
val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)  # For windows keep num_workers = 0


# Loss Functions
bce_loss = nn.BCEWithLogitsLoss()

# Check for Cuda availability
if torch.cuda.is_available():
    decive = 'cuda:0'
else:
    device = 'cpu'

# Initialization 
net = dnn(40, 1, 128, 64, 32).to(device)

# Initialize the optimizers
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


# Training Function
def training(data_loader, n_epochs):
    net.train()
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).to(device)

        optimizer.zero_grad()
        out = net(a)
        loss = bce_loss(out, b)
        
        loss.backward()
        optimizer.step()

        print ("[Epoch: %d] [Iter: %d/%d] [Loss: %f]" % (n_epochs, en, len(data_loader), loss.cpu().data.numpy()))
    

# Validation function
def validating(data_loader):
    net.eval()
    running_loss = 0
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).to(device)

        out = net(a)
        loss = bce_loss(out, b)

        running_loss += loss.item()
        
    return running_loss/(en+1)



def do_training():
    epoch = 5
    dl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%5==0:
            torch.save(net, join(checkpoint,"net_Ep_{}.pth".format(ep+1)))
        
        dl = validating(val_dataloader)
        print("loss: " + str(dl))
        dl_arr.append(dl)


        if ep == 0:
            dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='DNN'))
        else:
            viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    savemat(checkpoint+"/"+str('loss.mat'),  mdict={'foo': dl_arr})

    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(checkpoint+'/loss.png')



'''
Testing on training dataset as of now. Later it will be modified according to the different shell scripts.
'''


def do_testing():
    print("Testing")
    save_folder = "../results/mask/vuv"
    test_folder_path="../results/mask/mcc"  # Change the folder path to testing directory. (Later)
    dirs = listdir(test_folder_path)
    net = torch.load(join(checkpoint,"net_Ep_5.pth")).to(device)

    for i in dirs:
        
        # Load the .mat file
        d = read_mat(join(test_folder_path, i))

        a = torch.from_numpy(d['foo'])
        a = Variable(a.type('torch.FloatTensor')).to(device)
        
        Gout = net(a)

        savemat(join(save_folder,'{}.mat'.format(i[:-4])),  mdict={'foo': Gout.cpu().data.numpy()})


if __name__ == '__main__':
    do_training()
    do_testing()