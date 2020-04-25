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
    enc_nam.train()
    enc_whp.train()
    enc_sph.train()

    dec_nam.train()
    dec_whp.train()
    dec_sph.train()

    Dnet_whp.train()

    for en, (a, b, c, d) in enumerate(data_loader):

        a = Variable(a.squeeze(0).type(torch.FloatTensor)).cuda()
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).cuda()
        c = Variable(c.squeeze(0).type(torch.FloatTensor)).cuda()
        d = Variable(d.squeeze(0).type(torch.FloatTensor)).cuda()

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        ''' a - NAM | b - WHISPER-NAM | c - WHISPER-SPEECH | d - SPEECH, Here, WHISPER-NAM represents Whisper speech corresponding to NAM speech and 
        WHISPER-SPEECH represents Whisper speech  corresponding to Normal Speech'''

        ############# Generator ##############

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        enc_n = enc_nam(a)
        enc_w_n = enc_whp(b)
        enc_w_s = enc_whp(c)
        enc_s = enc_sph(d)

        a01 = dec_nam(enc_n)
        a02 = dec_nam(enc_w_n)

        bn_01 = dec_whp(enc_w_n)
        bn_02 = dec_whp(enc_n)

        bs_01 = dec_whp(enc_w_s)
        bs_02 = dec_whp(enc_s)

        c01 = dec_sph(enc_s)
        c02 = dec_sph(enc_w_s)

        # Losses for nam-whp
        loss1 = (adversarial_loss(a01,a) + adversarial_loss(a02,a))/2
        loss2 = (adversarial_loss(bn_01,b) + adversarial_loss(bn_02,b))/2

        # Losses for whp-sph
        loss3 = (adversarial_loss(bs_01,b) + adversarial_loss(bs_02,b))/2
        loss4 = (adversarial_loss(c01,c) + adversarial_loss(c02,c))/2

        loss5 = (adversarial_loss(enc_n, enc_w_n) + adversarial_loss(enc_w_s, enc_s))/2

        fake_w = (bce(Dnet_whp(bn_02.detach()), valid) + bce(Dnet_whp(bs_02.detach()), valid))/2

        autoencoder_loss = (loss1*10 + loss2*10 + loss3 + loss4 + loss5) + fake_w

        autoencoder_loss.backward(retain_graph=True)

        optimizer_enc.step()
        optimizer_dec.step()

        ############# Discriminator ###############

        optimizer_D_w.zero_grad()

        loss_D_real_n = bce(Dnet_nam(a01), valid) 
        loss_D_fake_n = bce(Dnet_nam(a02), fake) 

        loss_D_real_w = (bce(Dnet_whp(bn_01.detach()), valid) + bce(Dnet_whp(bs_01.detach()), valid))/2
        loss_D_fake_w = (bce(Dnet_whp(bn_02.detach()), fake) + bce(Dnet_whp(bs_02.detach()), fake))/2

        loss_D_real_s = bce(Dnet_sph(c01.detach()), valid)
        loss_D_fake_s = bce(Dnet_sph(c02.detach()), fake)

        Dnet_whp_loss = (loss_D_real_w + loss_D_fake_w)/2
        Dnet_whp_loss.backward()
        optimizer_D_w.step()

        print ("[Epoch: %d] [Iter:%d/%d] [Autoen: %f] [Dis_wph: %f]"% (n_epochs, en, len(data_loader), autoencoder_loss.cpu().data.numpy(), Dnet_whp_loss.cpu().data.numpy()))
    

# Validation function
def validating(data_loader):
    Drunning_loss_nam = 0
    Drunning_loss_whp = 0
    Drunning_loss_sph = 0
    Autoerunning_loss = 0
    Frunning_loss = 0

    enc_nam.eval()
    enc_whp.eval()
    enc_sph.eval()

    dec_nam.eval()
    dec_whp.eval()
    dec_sph.eval()

    for en, (a, b, c,d) in enumerate(data_loader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).cuda()
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).cuda()
        c = Variable(c.squeeze(0).type(torch.FloatTensor)).cuda()
        d = Variable(d.squeeze(0).type(torch.FloatTensor)).cuda()

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        ''' a - NAM | b - WHISPER-NAM | c - WHISPER-SPEECH | d - SPEECH '''

        ###### Generator #########

        enc_n = enc_nam(a)
        enc_w_n = enc_whp(b)
        enc_w_s = enc_whp(c)
        enc_s = enc_sph(d)

        a01 = dec_nam(enc_n)
        a02 = dec_nam(enc_w_n)

        bn_01 = dec_whp(enc_w_n)
        bn_02 = dec_whp(enc_n)

        bs_01 = dec_whp(enc_w_s)
        bs_02 = dec_whp(enc_s)

        c01 = dec_sph(enc_s)
        c02 = dec_sph(enc_w_s)

       # Losses for nam-whp
        loss1 = (adversarial_loss(a01,a) + adversarial_loss(a02,a))/2
        loss2 = (adversarial_loss(bn_01,b) + adversarial_loss(bn_02,b))/2

        # Losses for whp-sph
        loss3 = (adversarial_loss(bs_01,b) + adversarial_loss(bs_02,b))/2
        loss4 = (adversarial_loss(c01,c) + adversarial_loss(c02,c))/2

        loss5 = (adversarial_loss(enc_n, enc_w_n) + adversarial_loss(enc_w_s, enc_s))/2

        fake_w = (bce(Dnet_whp(bn_02.detach()), valid) + bce(Dnet_whp(bs_02.detach()), valid))/2

        autoencoder_loss = (loss1*10 + loss2*10 + loss3 + loss4 + loss5)+ fake_w

        ############# Discriminator ##############

        loss_D_real_w = (bce(Dnet_whp(bn_01.detach()), valid) + bce(Dnet_whp(bs_01.detach()), valid))/2
        loss_D_fake_w = (bce(Dnet_whp(bn_02.detach()), fake) + bce(Dnet_whp(bs_02.detach()), fake))/2

        Dnet_whp_loss = (loss_D_real_w + loss_D_fake_w)/2

        #################################################################

        Autoerunning_loss += autoencoder_loss.item()

        Drunning_loss_whp += Dnet_whp_loss.item()

    return Autoerunning_loss/(en+1), Drunning_loss_whp/(en+1) 


def do_training():
    epoch = args.epoch

    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%args.checkpoint_interval==0:
            torch.save(enc_nam, join(checkpoint,"enc_nam_Ep_{}.pth".format(ep+1)))
            torch.save(enc_whp, join(checkpoint,"enc_nam_Ep_{}.pth".format(ep+1)))
            torch.save(enc_sph, join(checkpoint,"enc_nam_Ep_{}.pth".format(ep+1)))
            torch.save(dec_nam, join(checkpoint,"enc_nam_Ep_{}.pth".format(ep+1)))
            torch.save(dec_whp, join(checkpoint,"enc_nam_Ep_{}.pth".format(ep+1)))
            torch.save(dec_sph, join(checkpoint,"enc_nam_Ep_{}.pth".format(ep+1)))


        if (ep+1)%args.validation_interval==0:
            dl,gl = validating(val_dataloader)
            
            print("AE_loss: " + str(dl) + " D_loss: " + str(gl))
            
            dl_arr.append(dl)
            gl_arr.append(gl)
            
            if ep == 0:
                gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Discriminator'))
                dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Auto-Encoders'))
            else:
                viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
                viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    savemat(checkpoint+"/"+str('autoencoders_loss.mat'),  mdict={'foo': dl_arr})
    savemat(checkpoint+"/"+str('discriminator_loss.mat'),  mdict={'foo': gl_arr})

    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(checkpoint+'/autoencoders_loss.png')
    plt.figure(2)
    plt.plot(gl_arr)
    plt.savefig(checkpoint+'/discriminator_loss.png')



'''
Testing on training dataset as of now. Later it will be modified according to the different shell scripts.
'''


def do_testing():
    print("Testing")
    save_folder = args.save_folder
    test_folder_path=args.test_folder

    dirs = listdir(test_folder_path)


    if args.test_type == "whsp2spch":
        enc = torch.load(join(checkpoint,"enc_whp_Ep_{}.pth".format(args.test_epoch))).to(device)
        dec = torch.load(join(checkpoint,"dec_sph_Ep_{}.pth".format(args.test_epoch))).to(device)

    if args.test_type == "nam2spch":
        enc = torch.load(join(checkpoint,"enc_nam_Ep_{}.pth".format(args.test_epoch))).to(device)
        dec = torch.load(join(checkpoint,"dec_sph_Ep_{}.pth".format(args.test_epoch))).to(device)

    if args.test_type == "nam2whsp":
        enc = torch.load(join(checkpoint,"enc_nam_Ep_{}.pth".format(args.test_epoch))).to(device)
        dec = torch.load(join(checkpoint,"dec_whp_Ep_{}.pth".format(args.test_epoch))).to(device)

    for i in dirs:
        
        # Load the .mcc file
        d = read_mcc(join(test_folder_path, i))

        a = torch.from_numpy(d)
        a = Variable(a.squeeze(0).type('torch.FloatTensor')).to(device)
        
        Gout = dec(enc(a))
        savemat(join(save_folder,'{}.mat'.format(i[:-4])),  mdict={'foo': Gout.cpu().data.numpy()})


'''
Check MCD value on validation data for now! :)
'''


def give_MCD():
    
    enc1 = torch.load(join(checkpoint,"enc_whp_Ep_{}.pth".format(args.test_epoch))).to(device)
    dec1 = torch.load(join(checkpoint,"dec_sph_Ep_{}.pth".format(args.test_epoch))).to(device)

    enc2 = torch.load(join(checkpoint,"enc_nam_Ep_{}.pth".format(args.test_epoch))).to(device)
    dec2 = torch.load(join(checkpoint,"dec_sph_Ep_{}.pth".format(args.test_epoch))).to(device)

    enc3 = torch.load(join(checkpoint,"enc_nam_Ep_{}.pth".format(args.test_epoch))).to(device)
    dec3 = torch.load(join(checkpoint,"dec_whp_Ep_{}.pth".format(args.test_epoch))).to(device)


    mcd_whsp2spch = []
    mcd_nam2whsp = []

    print("As of now MCD calculation relies upon the available parallel data. Hence, here, we calculat MCD for whsp2spch and nam2whsp conversions.")

    for en, (a, b, c, d) in enumerate(val_dataloader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = b.cpu().data.numpy()[0]

        c = Variable(c.squeeze(0).type(torch.FloatTensor)).to(device)
        d = d.cpu().data.numpy()[0]

        Gout1 = dec3(enc3(a)).cpu().data.numpy()
        Gout2 = dec1(enc1(c)).cpu().data.numpy()

        ans = 0
        for k in range(Gout1.shape[0]):
            ans = logSpecDbDist(Gout1[k][1:],b[k][1:])
            mcd_nam2whsp.append(ans)
            ans = logSpecDbDist(Gout2[k][1:],b[k][1:])
            mcd_whsp2spch.append(ans)

    mcd_whsp2spch = np.array(mcd_whsp2spch)
    mcd_nam2whsp = np.array(mcd_nam2whsp)
    print("MCD Scores: WHSP2SPCH={}\tNAM2WHSP={}".format(np.mean(mcd_whsp2spch), np.mean(mcd_nam2whsp)))


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
