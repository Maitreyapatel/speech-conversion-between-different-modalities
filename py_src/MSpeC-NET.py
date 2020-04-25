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

from dataloaders import mspec_net_speech_data
from networks import dnn_encoder, dnn_decoder, dnn_discriminator
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
        
        # Identity loss
        # G_W2S(S) should equal S if real S is fed
        same_s = Gnet_ws(b)
        loss_identity_s = criterion_identity(same_s, b)*5.0
        # G_S2W(W) should equal W if real W is fed
        same_w = Gnet_sw(a)
        loss_identity_w = criterion_identity(same_w, a)*5.0

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
        loss_G =  loss_identity_w + loss_identity_s + loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS
        loss_G.backward()
        
        optimizer_G.step()

        
        
#        Gout = Gnet(a)
#        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)*10
#
#        G_loss.backward()
#        optimizer_G.step()
        
        
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
        
        

        # D_loss = 0

        #D_running_loss = 0
        #D_running_loss += D_loss.item()
        
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
        
        # Identity loss
        # G_W2S(S) should equal S if real S is fed
        same_s = Gnet_ws(b)
        loss_identity_s = criterion_identity(same_s, b)*5.0
        # G_S2W(W) should equal W if real W is fed
        same_w = Gnet_sw(a)
        loss_identity_w = criterion_identity(same_w, a)*5.0
        
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
        loss_G =  loss_identity_w + loss_identity_s + loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS

        
        
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
    print("Testing")
    save_folder = args.save_folder
    test_folder_path=args.test_folder

    dirs = listdir(test_folder_path)
    Gnet = torch.load(join(checkpoint,"gen_ws_Ep_{}.pth".format(args.test_epoch))).to(device)

    for i in dirs:
        
        # Load the .mcc file
        d = read_mcc(join(test_folder_path, i))

        a = torch.from_numpy(d)
        a = Variable(a.squeeze(0).type('torch.FloatTensor')).to(device)
        
        Gout = Gnet(a)

        savemat(join(save_folder,'{}.mat'.format(i[:-4])),  mdict={'foo': Gout.cpu().data.numpy()})


'''
Check MCD value on validation data for now! :)
'''


def give_MCD():
    Gnet = torch.load(join(checkpoint,"gen_ws_Ep_{}.pth".format(args.test_epoch))).to(device)
    mcd = []

    for en, (a, b) in enumerate(val_dataloader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = b.cpu().data.numpy()[0]

        Gout = Gnet(a).cpu().data.numpy()

        ans = 0
        for k in range(Gout.shape[0]):
            ans = logSpecDbDist(Gout[k][1:],b[k][1:])
            mcd.append(ans)

    mcd = np.array(mcd)
    print(np.mean(mcd))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Training methodology for Whisper-to-Normal Speech Conversion")
    parser.add_argument("-mn", "--mspecnet", type=bool, default=False, help="If one wants to train MSpeC-Net.")
    parser.add_argument("-np", "--nonparallel", type=bool, default=False, help="Parallel training or non-parallel?")
    parser.add_argument("-dc", "--dnn_cnn", type=str, default='dnn', help="DNN or CNN architecture for generator and discriminator?")
    parser.add_argument("-tr", "--train", action="store_true", help="Want to train?")
    parser.add_argument("-te", "--test", action="store_true", help="Want to test?")
    parser.add_argument("-m", "--mcd", action="store_true", help="Want MCD value?")
    parser.add_argument("-ci", "--checkpoint_interval", type=int, default=5, help="Checkpoint interval")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="Number of Epochs")
    parser.add_argument("-et", "--test_epoch", type=int, default=100, help="Epochs to test")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-vi", "--validation_interval", type=int, default=1, help="Validation Interval")
    parser.add_argument("-mf1", "--mainfolder1", type=str, default="../dataset/features/MSpeC-Net/NAM2WHSP/batches/mcc/", help="Main folder path to load NAM-Whisper MCC batches")
    parser.add_argument("-mf2", "--mainfolder2", type=str, default="../dataset/features/MSpeC-Net/WHSP2SPCH/batches/mcc/", help="Main folder path to load Whisper-Normal Speech MCC batches")
    parser.add_argument("-cf", "--checkpoint_folder", type=str, default="../results/checkpoints/mcc/", help="Checkpoint saving path for MCC features")
    parser.add_argument("-sf", "--save_folder", type=str, default="../results/mask/mcc/", help="Saving folder for converted MCC features")
    parser.add_argument("-tt", "--test_type", type=str, default="whsp2spch", help="Provide the type of conversation to be tested out.")
    parser.add_argument("-tf", "--test_folder", type=str, default="../dataset/features/MSpeC-Net/Whisper/mcc/", help="Input whisper mcc features for testing")

    args = parser.parse_args()


    
    # Connect with Visdom for the loss visualization
    viz = visdom.Visdom()

    # Path where you want to store your results        
    mainfolder1 = args.mainfolder1
    mainfolder2 = args.mainfolder2
    checkpoint = args.checkpoint_folder

    # Training Data path
    if args.nonparallel:
        print("Currently, MSpeC-Net does not support non-parallel training")
    
    custom_dataloader = mspec_net_speech_data

    traindata = custom_dataloader(folder1=mainfolder1, folder2=mainfolder2)
    train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=0)  # For windows keep num_workers = 0


    # Path for validation data
    valdata = custom_dataloader(folder1=mainfolder1, folder2=mainfolder2)
    val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=0)  # For windows keep num_workers = 0


    # Loss Functions
    bce = nn.BCELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    adversarial_loss = nn.MSELoss()

    #I/O variables
    in_enc = 40         # Input dimension of encoder
    out_enc = 512       # Output dimension for encoder - latent space

    in_dec = 512        # Input dimension for decoder - latent space
    out_dec = 40        # Output dimension of decoder

    in_d = 40           # Discriminator input dim
    out_d = 1           # Discriminator output dim



    # Check for Cuda availability
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Initialization
    if args.dnn_cnn == "cnn":
        print("Currently, MSpeC-Net only supports DNN based architectures.")

    enc_nam = dnn_encoder(in_enc, out_enc, 512, 512, 512).cuda()
    enc_whp = dnn_encoder(in_enc, out_enc, 512, 512, 512).cuda()
    enc_sph = dnn_encoder(in_enc, out_enc, 512, 512, 512).cuda()

    dec_nam = dnn_decoder(in_dec, out_dec, 512, 512, 512).cuda()
    dec_whp = dnn_decoder(in_dec, out_dec, 512, 512, 512).cuda()
    dec_sph = dnn_decoder(in_dec, out_dec, 512, 512, 512).cuda()

    Dnet_nam = dnn_discriminator(in_d, out_d, 512, 512, 512).cuda()
    Dnet_whp = dnn_discriminator(in_d, out_d, 512, 512, 512).cuda()
    Dnet_sph = dnn_discriminator(in_d, out_d, 512, 512, 512).cuda()



    # Initialize the optimizers
    params1 = list(enc_nam.parameters()) + list(enc_whp.parameters()) + list(enc_sph.parameters())
    params2 = list(dec_nam.parameters()) + list(dec_whp.parameters()) + list(dec_sph.parameters())
    optimizer_enc = torch.optim.Adam(params1, lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_dec = torch.optim.Adam(params2, lr=args.learning_rate, betas=(0.5, 0.999))

    optimizer_D_w = torch.optim.Adam(Dnet_whp.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    if args.train:
        do_training()
    if args.test:
        do_testing()
    if args.mcd:
        give_MCD()
