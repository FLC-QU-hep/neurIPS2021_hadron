import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

      
class DCGAN_G(nn.Module):
    """ 
        generator component of WGAN
    """
    def __init__(self, ngf, nz):
        super(DCGAN_G, self).__init__()
        
        self.ngf = ngf
        self.nz = nz

        kernel = 4
        
        # input energy shape [batch x 1 x 1 x 1 ] going into convolutional
        self.conv1_1 = nn.ConvTranspose3d(1, ngf*4, kernel, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 x 4]
        
        # input noise shape [batch x nz x 1 x 1] going into convolutional
        self.conv1_100 = nn.ConvTranspose3d(nz, ngf*4, kernel, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 x 4]
        
        
        # outs from first convolutions concatenate state size [ ngf*8 x 4 x 4]
        # and going into main convolutional part of Generator
        self.main_conv = nn.Sequential(
            
            nn.ConvTranspose3d(ngf*8, ngf*4, kernel_size=(4,2,2), stride=2, padding=1, bias=False),
            nn.LayerNorm([8, 6, 6]),
            nn.ReLU(),
            # state shape [ (ndf*4) x 6 x 6 ]

            nn.ConvTranspose3d(ngf*4, ngf*2, kernel_size=(4,2,2), stride=2, padding=1, bias=False),
            nn.LayerNorm([16, 10, 10]),
            nn.ReLU(),
            # state shape [ (ndf*2) x 10 x 10 ]

            nn.ConvTranspose3d(ngf*2, ngf, kernel_size=(4,4,4), stride=(2,1,1), padding=1, bias=False),
            nn.LayerNorm([32, 11, 11]),
            nn.ReLU(),
            # state shape [ (ndf) x 11 x 11 ]

            nn.ConvTranspose3d(ngf, 10, kernel_size=(10,4,4), stride=1, padding=1, bias=False),
            nn.LayerNorm([39, 12, 12]),
            nn.ReLU(),
            # state shape [ ch=10 x 12 x 12 ]
           
            nn.ConvTranspose3d(10, 5, kernel_size=(8,3,3), stride=(1,2,2), padding=1, bias=False),
            nn.LayerNorm([44, 23, 23]),
            nn.ReLU(),
            
            # state shape [ ch=5 x 23 x 23  ]
            
            nn.ConvTranspose3d(5, 1, kernel_size=(7,5,5), stride=1, padding=1, bias=False),
            nn.ReLU()
            
            ## final output ---> [48 x 25 x 25]
        )

    def forward(self, noise, energy):
        energy_trans = self.conv1_1(energy)
        noise_trans = self.conv1_100(noise)
        input = torch.cat((energy_trans, noise_trans), 1)
        x = self.main_conv(input)
        x = x.view(-1, 48, 25, 25)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

        
