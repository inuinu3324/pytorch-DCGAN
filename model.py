import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#parameter
noize_size = 100
ngf = 64
ndf = 64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.l0 = nn.Linear(noize_size, 4*4*ngf*8, bias = False)
        self.dconv0 = nn.ConvTranspose2d(ngf*8, ngf*4,4,2,1,bias = False)
        self.dconv1 = nn.ConvTranspose2d(ngf*4, ngf*2,4,2,1,bias = False)
        self.dconv2 = nn.ConvTranspose2d(ngf*2, ngf,4,2,1,bias = False)
        self.dconv3 = nn.ConvTranspose2d(ngf, 3,4,2,1,bias = False)

        self.bn0l = nn.BatchNorm1d(4*4*ngf*8)
        self.bn0 = nn.BatchNorm2d(ngf*4)
        self.bn1 = nn.BatchNorm2d(ngf*2)
        self.bn2 = nn.BatchNorm2d(ngf)
        
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = torch.reshape(F.relu(self.bn0l(self.l0(x))), (x.data.shape[0],ngf*8,4,4))
        h = self.relu(self.bn0(self.dconv0(h)))
        h = self.relu(self.bn1(self.dconv1(h)))
        h = self.relu(self.bn2(self.dconv2(h)))
        h = self.tanh(self.dconv3(h))
        return h

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.c0 = nn.Conv2d(3, ndf,4,2,1, bias = False)
        self.c1 = nn.Conv2d(ndf, ndf*2,4,2,1, bias = False)
        self.c2 = nn.Conv2d(ndf*2, ndf*4,4,2,1, bias = False)
        self.c3 = nn.Conv2d(ndf*4, ndf*8,4,2,1, bias = False)
        self.l4l = nn.Linear(4*4*ndf*8,2)
        
        #self.bn0 = nn.BatchNorm2d(ndf)
        self.bn1 = nn.BatchNorm2d(ndf*2)
        self.bn2 = nn.BatchNorm2d(ndf*4)
        self.bn3 = nn.BatchNorm2d(ndf*8)
        
        self.act0 = nn.LeakyReLU(0.2, inplace=True)
        self.act1 = nn.Sigmoid()
        
    def forward(self, x):
        h = self.act0(self.c0(x))
        h = self.act0(self.bn1(self.c1(h)))
        h = self.act0(self.bn2(self.c2(h)))
        h = self.act0(self.bn3(self.c3(h)))
        h = torch.reshape(h,(x.data.shape[0],-1))
        h = self.act1(self.l4l(h))
        return h

if __name__ == "__main__":  
    netG = Generator(1)
    netG.apply(weights_init)

    z = torch.randn(64,noize_size)
    print(z.data.shape)

    ge = netG(z)

    netD = Discriminator(1)
    netD.apply(weights_init)

    z = torch.randn(128,3,64,64)
    print(z.data.shape)

    gd = netD(z)
    print(gd.shape)