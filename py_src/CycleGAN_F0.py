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
import itertools

import visdom
import math
import matplotlib.pyplot as plt 
import scipy
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

from dataloaders import parallel_dataloader, non_parallel_dataloader
from networks import dnn_generator, dnn_discriminator
from utils import *

import argparse



# Training Function
def training(data_loader, n_epochs):
    Gnet_ws.train()
    Gnet_sw.train()
    Dnet_w.train()
    Dnet_s.train()
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).to(device)

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        
        ###### Generators W2S and S2W ######
        optimizer_G.zero_grad()
        
        # GAN loss
        Gout_ws = Gnet_ws(a)
        loss_GAN_W2S = criterion_GAN(Dnet_s(Gout_ws), valid)
        
        Gout_sw = Gnet_sw(b)
        loss_GAN_S2W = criterion_GAN(Dnet_w(Gout_sw), valid)
        
        # Cycle loss
        recovered_W = Gnet_sw(Gout_ws)
        loss_cycle_WSW = criterion_cycle(recovered_W, a)*10.0
        
        recovered_S = Gnet_ws(Gout_sw)
        loss_cycle_SWS = criterion_cycle(recovered_S, b)*10.0
        
        # Total loss
        loss_G = loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS
        loss_G.backward()
        
        optimizer_G.step()

        
        
        ###### Discriminator W ######
        optimizer_D_w.zero_grad()

        # Real loss
        loss_D_real = criterion_GAN(Dnet_w(a), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_w(Gout_sw.detach()), fake)
        
        # Total loss
        loss_D_w = (loss_D_real + loss_D_fake)*0.5
        loss_D_w.backward()
        
        optimizer_D_w.step()
        
        ###################################
        
        ###### Discriminator B ######
        optimizer_D_s.zero_grad()
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_s(b), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_s(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_s = (loss_D_real + loss_D_fake)*0.5
        loss_D_s.backward()
        
        optimizer_D_s.step()
        ###################################
        
        print ("[Epoch: %d] [Iter: %d/%d] [D_S loss: %f] [D_W loss: %f] [G loss: %f]" % (n_epochs, en, len(data_loader), loss_D_s, loss_D_w, loss_G.cpu().data.numpy()))
    

# Validation function
def validating(data_loader):
    Gnet_ws.eval()
    Gnet_sw.eval()
    Dnet_s.eval()
    Dnet_w.eval()
    Grunning_loss = 0
    Drunning_loss = 0
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).to(device)

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        
        ###### Generators W2S and S2W ######
        
        # GAN loss
        Gout_ws = Gnet_ws(a)
        loss_GAN_W2S = criterion_GAN(Dnet_s(Gout_ws), valid)
        
        Gout_sw = Gnet_sw(b)
        loss_GAN_S2W = criterion_GAN(Dnet_w(Gout_sw), valid)
        
        # Cycle loss
        recovered_W = Gnet_sw(Gout_ws)
        loss_cycle_WSW = criterion_cycle(recovered_W, a)*10.0
        
        recovered_S = Gnet_ws(Gout_sw)
        loss_cycle_SWS = criterion_cycle(recovered_S, b)*10.0
        
        # Total loss
        loss_G = loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS

        
        
        ###### Discriminator W ######
        optimizer_D_w.zero_grad()
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_w(a), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_w(Gout_sw.detach()), fake)
        
        # Total loss
        loss_D_w = (loss_D_real + loss_D_fake)*0.5

        
        ###### Discriminator B ######
        optimizer_D_s.zero_grad()
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_s(b), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_s(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_s = (loss_D_real + loss_D_fake)*0.5
 
        ###################################
        loss_D = loss_D_s + loss_D_w	

        Grunning_loss += loss_G.item()

        Drunning_loss += loss_D.item()
        
    return Drunning_loss/(en+1),Grunning_loss/(en+1)
    

def do_training():
    epoch = args.epoch
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%args.checkpoint_interval==0:
            torch.save(Gnet_ws, join(checkpoint,"gen_ws_Ep_{}.pth".format(ep+1)))
        
        if (ep+1)%args.validation_interval==0:
            dl,gl = validating(val_dataloader)
            print("D_loss: " + str(dl) + " G_loss: " + str(gl))
            dl_arr.append(dl)
            gl_arr.append(gl)
            
            if ep == 0:
                gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Generator'))
                dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Discriminator'))
            else:
                viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
                viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    savemat(checkpoint+"/"+str('discriminator_loss.mat'),  mdict={'foo': dl_arr})
    savemat(checkpoint+"/"+str('generator_loss.mat'),  mdict={'foo': gl_arr})

    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(checkpoint+'/discriminator_loss.png')
    plt.figure(2)
    plt.plot(gl_arr)
    plt.savefig(checkpoint+'/generator_loss.png')



'''
Testing on training dataset as of now. Later it will be modified according to the different shell scripts.
'''


def do_testing():
    save_folder = args.save_folder
    test_folder_path = args.test_folder
    dirs = listdir(test_folder_path)
    Gnet = torch.load(join(checkpoint,"gen_ws_Ep_{}.pth".format(args.test_epoch))).to(device)

    for i in dirs:
        
        # Load the .mat file
        d = read_mat(join(test_folder_path, i))

        a = torch.from_numpy(d['foo'])
        a = Variable(a.type('torch.FloatTensor')).to(device)
        
        Gout = Gnet(a)

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
    parser.add_argument("-mf", "--mainfolder", type=str, default="../dataset/features/US_102/batches/f0/", help="Main folder path to load F0 batches")
    parser.add_argument("-cf", "--checkpoint_folder", type=str, default="../results/checkpoints/f0/", help="Checkpoint saving path for F0 features")
    parser.add_argument("-sf", "--save_folder", type=str, default="../results/mask/f0/", help="Saving folder for converted MCC features")
    parser.add_argument("-tf", "--test_folder", type=str, default="../results/mask/mcc/", help="Input whisper mcc features for testing")

    args = parser.parse_args()




    # Connect with Visdom for the loss visualization
    viz = visdom.Visdom()

    # Path where you want to store your results        
    mainfolder = args.mainfolder
    checkpoint = args.checkpoint_folder

    # Training Data path
    if args.nonparallel:
        custom_dataloader = non_parallel_dataloader
    else:
        custom_dataloader = parallel_dataloader


    traindata = custom_dataloader(folder_path=mainfolder)
    train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)  # For windows keep num_workers = 0


    # Path for validation data
    valdata = custom_dataloader(folder_path=mainfolder)
    val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)  # For windows keep num_workers = 0


    # Loss Functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    #I/O variables
    in_g = 40
    out_g = 40
    in_d = 40
    out_d = 1


    # Check for Cuda availability
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Initialization 
    if args.dnn_cnn == "dnn":
        Gnet_ws = dnn_generator(in_g, 1, 512, 512, 512).to(device)
        Gnet_sw = dnn_generator(1, out_g, 512, 512, 512).to(device)
        Dnet_w = dnn_discriminator(in_d, out_d, 512, 512, 512).to(device)
        Dnet_s = dnn_discriminator(1, out_d, 512, 512, 512).to(device)


    # Initialize the optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(Gnet_ws.parameters(), Gnet_sw.parameters()),lr=args.learning_rate)
    optimizer_D_w = torch.optim.Adam(Dnet_w.parameters(), lr=args.learning_rate)
    optimizer_D_s = torch.optim.Adam(Dnet_s.parameters(), lr=args.learning_rate)


    if args.train:
        do_training()
    if args.test:
        do_testing()