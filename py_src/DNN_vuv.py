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

from dataloaders import parallel_dataloader, non_parallel_dataloader
from networks import dnn
from utils import *

import argparse



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
    epoch = args.epoch
    dl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%args.checkpoint_interval==0:
            torch.save(net, join(checkpoint,"net_Ep_{}.pth".format(ep+1)))
        
        if (ep+1)%args.validation_interval==0:
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
    save_folder = args.save_folder
    test_folder_path = args.test_folder
    dirs = listdir(test_folder_path)
    net = torch.load(join(checkpoint,"net_Ep_{}.pth".format(args.test_epoch))).to(device)

    for i in dirs:
        
        # Load the .mat file
        d = read_mat(join(test_folder_path, i))

        a = torch.from_numpy(d['foo'])
        a = Variable(a.type('torch.FloatTensor')).to(device)
        
        Gout = net(a)

        savemat(join(save_folder,'{}.mat'.format(i[:-4])),  mdict={'foo': Gout.cpu().data.numpy()})


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training methodology for Whisper-to-Normal Speech Conversion")
    parser.add_argument("-np", "--nonparallel", type=bool, default=False, help="Parallel training or non-parallel?")
    parser.add_argument("-dc", "--dnn_cnn", type=str, default='dnn', help="DNN or CNN architecture for generator and discriminator?")
    parser.add_argument("-tr", "--train", action="store_true", help="Want to train?")
    parser.add_argument("-te", "--test", action="store_true", help="Want to test?")
    parser.add_argument("-ci", "--checkpoint_interval", type=int, default=5, help="Checkpoint interval")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="Number of Epochs")
    parser.add_argument("-et", "--test_epoch", type=int, default=100, help="Epochs to test")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-vi", "--validation_interval", type=int, default=1, help="Validation Interval")
    parser.add_argument("-mf", "--mainfolder", type=str, default="../dataset/features/US_102/batches/VUV/", help="Main folder path to load VUV batches")
    parser.add_argument("-cf", "--checkpoint_folder", type=str, default="../results/checkpoints/vuv/", help="Checkpoint saving path for VUV features")
    parser.add_argument("-sf", "--save_folder", type=str, default="../results/mask/vuv/", help="Saving folder for converted MCC features")
    parser.add_argument("-tf", "--test_folder", type=str, default="../results/mask/mcc/", help="Input whisper mcc features for testing")

    args = parser.parse_args()

    
    # Connect with Visdom for the loss visualization
    viz = visdom.Visdom()

    # Path where you want to store your results        
    mainfolder = args.mainfolder
    checkpoint = args.checkpoint_folder

    if args.nonparallel:
        custom_dataloader = non_parallel_dataloader
    else:
        custom_dataloader = parallel_dataloader

    # Training Data path
    traindata = custom_dataloader(folder_path=mainfolder)
    train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)  # For windows keep num_workers = 0


    # Path for validation data
    valdata = custom_dataloader(folder_path=mainfolder)
    val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)  # For windows keep num_workers = 0


    # Loss Functions
    bce_loss = nn.BCEWithLogitsLoss()

    # Check for Cuda availability
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    # Initialization 
    if args.dnn_cnn == "dnn":
        net = dnn(40, 1, 128, 64, 32).to(device)

    # Initialize the optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    if args.train:
        do_training()
    if args.test:
        do_testing()