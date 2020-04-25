'''
Here, the generators and discriminators are pre-defined as per the configuration used in research paper.
'''
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


# Generator (Converts Whisper-to-Speech) consists of DNN
class dnn_generator(nn.Module):
    
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.out.bias)

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(dnn_generator, self).__init__()
        
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, G_out)

        #self.weight_init()
    
    # Deep neural network [you are passing data layer-to-layer]    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x
        


# Discriminator (Dicriminate between Whispered speeches) also consists of DNN
class dnn_discriminator(nn.Module):

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)
    
    def __init__(self, D_in, D_out, w1, w2, w3):
        super(dnn_discriminator, self).__init__()
        
        self.fc1= nn.Linear(D_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, D_out)
        
        #self.weight_init()
        
    def forward(self, y):
        
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.sigmoid(self.out(y))
        return y


class dnn_encoder(nn.Module):

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(dnn_encoder, self).__init__()

        self.fc1 = nn.Linear(G_in, w1)
        self.fc2 = nn.Linear(w1, w2)
        self.fc3 = nn.Linear(w2, w3)
        self.out = nn.Linear(w3, G_out)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.out(x)
        return x

class dnn_decoder(nn.Module):

    def __init__(self, D_in, D_out, w1, w2, w3):

        super(dnn_decoder, self).__init__()

        self.fc1 = nn.Linear(D_in, w1)
        self.fc2 = nn.Linear(w1, w2)
        self.fc3 = nn.Linear(w2, w3)
        self.out = nn.Linear(w3, D_out)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x



class dnn(nn.Module):
    
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(dnn, self).__init__()
        
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, G_out)

        # self.weight_init()
        
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.out(x))
        # x = x.view(1, 1, 1000, 25)
        return x


class cnn_generator(nn.Module):
    
    
    def __init__(self):
        super(cnn_generator, self).__init__()
        
        lower_layers = []
        self.lower_layers = nn.Sequential(*lower_layers)
        
        inception_layers = []
        inception_layers += [nn.Conv2d(1, 128, 3, padding=1)]
        inception_layers += [nn.ReLU(True)]
        inception_layers += [nn.Conv2d(128, 256, 3, padding=1)]
        inception_layers += [nn.ReLU(True)]
        inception_layers += [nn.Conv2d(256, 128, 3, padding=1)]
        inception_layers += [nn.ReLU(True)]
        self.inception_layers = nn.Sequential(*inception_layers)

        final_layers = []
        final_layers += [nn.Conv2d(128, 1, 3, stride=1, padding=1)] # Out: 1000x40x1
        self.final_layers = nn.Sequential(*final_layers)

        
    def forward(self, x):
        
        h1 = self.lower_layers(x)
        h2 = self.inception_layers(h1)
        return self.final_layers(h2)

class cnn_f0_generator(nn.Module):
    
    
    def __init__(self):
        super(cnn_f0_generator, self).__init__()
        
        lower_layers = []
        self.lower_layers = nn.Sequential(*lower_layers)
        
        inception_layers = []
        inception_layers += [nn.Conv2d(1, 128, 3, padding=1)]
        inception_layers += [nn.ReLU(True)]
        inception_layers += [nn.MaxPool2d((1,7),(1,2))]
        inception_layers += [nn.Conv2d(128, 256, 3, padding=1)]
        inception_layers += [nn.ReLU(True)]
        inception_layers += [nn.MaxPool2d((1,7),(1,2))]
        inception_layers += [nn.Conv2d(256, 128, 3, padding=1)]
        inception_layers += [nn.ReLU(True)]
        inception_layers += [nn.MaxPool2d((1,6),(1,2))]
        self.inception_layers = nn.Sequential(*inception_layers)

        final_layers = []
        final_layers += [nn.Conv2d(128, 1, 3, stride=1, padding=1)]
        self.final_layers = nn.Sequential(*final_layers)

        
    def forward(self, x):
        
        h1 = self.lower_layers(x)
        h2 = self.inception_layers(h1)
        return self.final_layers(h2)   

class cnn_discriminator(nn.Module):
    
    def __init__(self):
        super(cnn_discriminator, self).__init__()
        
        lower_layers = []
        lower_layers += [nn.Conv2d(1, 32, 7, 2, 3)] # Out: 500x20x32
        lower_layers += [nn.ReLU(True)]
        lower_layers += [nn.Conv2d(32, 64, 3, 1, 1)] # Out: 1000x25x64
        lower_layers += [nn.ReLU(True)]
        lower_layers += [nn.MaxPool2d(3, (3,2), 1)] # Out: 500x13x192
        lower_layers += [nn.Conv2d(64, 128, 3, 1, 1)] # Out: 1000x25x192
        lower_layers += [nn.ReLU(True)]
        lower_layers += [nn.MaxPool2d(3, (4,2), 1)] # Out: 500x13x192
        lower_layers += [nn.Conv2d(128, 256, 3, 1, 1)] # Out: 1000x25x192
        lower_layers += [nn.ReLU(True)]
        lower_layers += [nn.MaxPool2d(3, (3,2), 1)] # Out: 500x13x192
        lower_layers += [nn.Conv2d(256, 256, 3, 1, 1)] # Out: 1000x25x192
        lower_layers += [nn.ReLU(True)]
        
        
        self.lower_layers = nn.Sequential(*lower_layers)
        
        
        final_layers = []
        
        final_layers += [nn.Linear(3*14*256, 1028)]
        final_layers += [nn.ReLU(True)]
        final_layers += [nn.Dropout(0.5)]
        final_layers += [nn.Linear(1028, 1)]
        final_layers += [nn.Sigmoid()]
        
        nn.init.xavier_uniform_(final_layers[0].weight)
        nn.init.xavier_uniform_(final_layers[3].weight)

        self.final_layers = nn.Sequential(*final_layers)
        
    def forward(self, x):
        h1 = self.lower_layers(x)
        h2 = h1.view(h1.size(0), -1)
        return self.final_layers(h2)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x



class inception(nn.Module):
    
    def __init__(self, inp, n1, n3r, n3, n5r, n5, mxp):
        super(inception, self).__init__()
        
        layers = []
        layers += [nn.Conv2d(inp, n1, 1)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[0].weight)
        
        self.one = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(inp, n3r, 1)]
        layers += [nn.ReLU(True)]
        layers += [nn.Conv2d(n3r, n3, 3, padding=1)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[0].weight)
        # nn.init.xavier_uniform_(layers[2].weight)
        
        self.three = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(inp, n5r, 1)]
        layers += [nn.ReLU(True)]
        layers += [nn.Conv2d(n5r, n5, 3, padding=1)]
        layers += [nn.ReLU(True)]
        layers += [nn.Conv2d(n5, n5, 3, padding=1)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[0].weight)
        # nn.init.xavier_uniform_(layers[2].weight)
        # nn.init.xavier_uniform_(layers[4].weight)
        
        self.five = nn.Sequential(*layers)
        
        layers = []
        
        layers += [nn.MaxPool2d(3, 1, 1)]
        layers += [nn.Conv2d(inp, mxp, 1)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[1].weight)
        
        self.maxp = nn.Sequential(*layers)
        
    def forward(self, x):
        h1 = self.one(x)
        h2 = self.three(x)
        h3 = self.five(x)
        h4 = self.maxp(x)
        
        h = torch.cat([h1, h2, h3, h4], 1)
        
        return h


class inv_inception(nn.Module):
    
    def __init__(self, inp, n1, n3r, n3, n5r, n5, mxp):
        super(inv_inception, self).__init__()
        
        layers = []
        layers += [nn.ConvTranspose2d(inp, n1, 1, stride=1, padding=0, output_padding=0)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[0].weight)
        
        self.one = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.ConvTranspose2d(inp, n3r, 1, stride=1, padding=0, output_padding=0)]
        layers += [nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(n3r, n3, 3, stride=1, padding=1, output_padding=0)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[0].weight)
        # nn.init.xavier_uniform_(layers[2].weight)
        
        self.three = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.ConvTranspose2d(inp, n5r, 1, stride=1, padding=0, output_padding=0)]
        layers += [nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(n5r, n5, 3, stride=1, padding=1, output_padding=0)]
        layers += [nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(n5, n5, 3, stride=1, padding=1, output_padding=0)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[0].weight)
        # nn.init.xavier_uniform_(layers[2].weight)
        # nn.init.xavier_uniform_(layers[4].weight)
        
        self.five = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.MaxPool2d(3, 1, 1)]
        layers += [nn.ConvTranspose2d(inp, mxp, 1, stride=1, padding=0, output_padding=0)]
        layers += [nn.ReLU(True)]
        # nn.init.xavier_uniform_(layers[1].weight)
        
        self.maxp = nn.Sequential(*layers)
        
    def forward(self, x):
        h1 = self.one(x)
        h2 = self.three(x)
        h3 = self.five(x)
        h4 = self.maxp(x)
        
        h = torch.cat([h1, h2, h3, h4], 1)
        
        return h


class inception_generator(nn.Module):
    
    
    def __init__(self):
        super(inception_generator, self).__init__()
        
        lower_layers = []
        self.lower_layers = nn.Sequential(*lower_layers)
        
        inception_layers = []
        inception_layers += [inception(1, 32, 50, 64, 16, 32, 16)] # Out: 500x20x144
        inception_layers += [inception(144, 64, 64, 128, 32, 64, 32)] # Out: 500x20x288
        inception_layers += [inv_inception(288, 64, 64, 128, 32, 64, 32)] # Out: 500x20x288  ## Do enable if you want to use inverse inception
        inception_layers += [inv_inception(288, 32, 128, 64, 64, 32, 16)] # Out: 500x20x144  ## Do chnage inception to inv_inception only keep the input values as it is if you want to use inverse inception
        self.inception_layers = nn.Sequential(*inception_layers)

        final_layers = []
        final_layers += [nn.Conv2d(144, 1, 3, stride=1, padding=1)] # Out: 1000x40x1
        self.final_layers = nn.Sequential(*final_layers)

        
    def forward(self, x):
        
        h1 = self.lower_layers(x)
        h2 = self.inception_layers(h1)
        return self.final_layers(h2)

class inception_f0_generator(nn.Module):
    
    
    def __init__(self):
        super(inception_f0_generator, self).__init__()
        
        lower_layers = []
        self.lower_layers = nn.Sequential(*lower_layers)
        
        inception_layers = []
        inception_layers += [inception(1, 32, 50, 64, 16, 32, 16)] # Out: 500x20x144
        inception_layers += [nn.MaxPool2d((1,7),(1,2))]
        inception_layers += [inception(144, 64, 64, 128, 32, 64, 32)] # Out: 500x20x288
        inception_layers += [nn.MaxPool2d((1,7),(1,2))]
        inception_layers += [inception(288, 64, 64, 128, 32, 64, 32)] # Out: 500x20x288  ## Do enable if you want to use inverse inception
        inception_layers += [nn.MaxPool2d((1,6),(1,2))]
        inception_layers += [inception(288, 32, 128, 64, 64, 32, 16)] # Out: 500x20x144  ## Do chnage inception to inv_inception only keep the input values as it is if you want to use inverse inception
        self.inception_layers = nn.Sequential(*inception_layers)

        final_layers = []
        final_layers += [nn.Conv2d(144, 1, 3, stride=1, padding=1)] # Out: 1000x40x1
        self.final_layers = nn.Sequential(*final_layers)

        
    def forward(self, x):
        
        h1 = self.lower_layers(x)
        h2 = self.inception_layers(h1)
        return self.final_layers(h2)



class inception_discriminator(nn.Module):
    
    def __init__(self):
        super(inception_discriminator, self).__init__()
        
        lower_layers = []
        lower_layers += [nn.Conv2d(1, 32, 7, 2, 3)] # Out: 500x20x32
        lower_layers += [nn.ReLU(True)]
        
        self.lower_layers = nn.Sequential(*lower_layers)
        
        inception_layers = []
        
        inception_layers += [inception(32, 32, 50, 64, 16, 32, 16)] # Out: 500x20x144
        inception_layers += [nn.AvgPool2d((25, 2), (5, 1))] # Out: 96x19x160

        inception_layers += [inception(144, 64, 64, 128, 32, 64, 32)] # Out: 96x19x288
        inception_layers += [nn.MaxPool2d(3, 2, 1)] # Out: 48x10x288
      
        inception_layers += [inception(288, 128, 128, 256, 32, 64, 48)]#out: 48x6x496
        inception_layers += [nn.AvgPool2d((10,2),(6,6))]#out: 7*2*496
        # inception_layers += [Print()]
        
        self.inception_layers = nn.Sequential(*inception_layers)
        
        final_layers = []
        
        final_layers += [nn.Linear(7*2*496, 1028)]
        final_layers += [nn.ReLU(True)]
        final_layers += [nn.Dropout(0.5)]
        final_layers += [nn.Linear(1028, 1)]
        final_layers += [nn.Sigmoid()]
        
        nn.init.xavier_uniform_(final_layers[0].weight)
        nn.init.xavier_uniform_(final_layers[3].weight)

        self.final_layers = nn.Sequential(*final_layers)
        
    def forward(self, x):
        h1 = self.lower_layers(x)
        h2 = self.inception_layers(h1)
        h2 = h2.view(h2.size(0), -1)
        return self.final_layers(h2)
