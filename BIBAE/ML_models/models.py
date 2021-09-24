import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import autograd           
    
class Latent_Critic(nn.Module):
    def __init__(self, ):
        super(Latent_Critic, self).__init__()
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 100)        
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 1)

    def forward(self, x):      
        x = F.leaky_relu(self.linear1(x.view(-1,1)), inplace=True)
        x = F.leaky_relu(self.linear2(x), inplace=True)
        x = F.leaky_relu(self.linear3(x), inplace=True)
        return self.linear4(x)
     
    
class Discriminator_F_BatchStat_Core25(nn.Module):
    def __init__(self, isize=48, nc=2, ndf=64):
        super(Discriminator_F_BatchStat_Core25, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.size_embed = 16
        self.conv1_bias = False

        self.Embed1a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed1c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed2a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed2c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed3a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed3c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)

        self.Embed4a = torch.nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4b = torch.nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        self.Embed4c = torch.nn.Conv1d(in_channels = 64, out_channels = self.size_embed, kernel_size = 1, 
                                     stride=1, padding = 0, bias = self.conv1_bias)
        
        self.embed_lin = torch.nn.Linear(self.size_embed*4, 64) 
        self.embed_log = torch.nn.Linear(self.size_embed*4, 64) 
        self.diff_lin = torch.nn.Linear(48*25*25, 64) 
        self.diff_log = torch.nn.Linear(48*25*25, 64) 
        
        
        self.conv1a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln1a = torch.nn.LayerNorm([24,24,24])
        self.conv1b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln1b = torch.nn.LayerNorm([12,12,12])
        self.conv1c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)
        
        self.conv2a = torch.nn.Conv3d(1, ndf, kernel_size=2, stride=(2,1,1), padding=0, bias=False)
        self.ln2a = torch.nn.LayerNorm([24,24,24])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=2, stride=(2,2,2), padding=0, bias=False)
        self.ln2b = torch.nn.LayerNorm([12,12,12])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=(2,2,2), padding=(1,1,1), bias=False)

        self.conv_lin = torch.nn.Linear(6*6*6*ndf, 64) 
        self.conv_log = torch.nn.Linear(6*6*6*ndf, 64) 
        
        self.econd_lin = torch.nn.Linear(1, 64) 

        self.fc1 = torch.nn.Linear(64*7, 256)
        self.fc2 = torch.nn.Linear(256,  256)
        self.fc3 = torch.nn.Linear(256, 1)


    def forward(self, img, E_true):
        batch_size = img.size(0)
        
        
        img_a = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        img_b = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        
        img_log_a = torch.log(img_a+np.exp(-5.0))+5.0
        img_log_b = torch.log(img_b+np.exp(-5.0))+5.0
        
        
        img_a_hit_std = torch.std(img_a.view(batch_size, -1), 1)
        img_a_hit_std_a = img_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_std_b = img_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_std = img_a_hit_std_a - img_a_hit_std_b
        
        img_a_hit_sum = torch.sum(img_a.view(batch_size, -1), 1)
        img_a_hit_sum_a = img_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_a_hit_sum_b = img_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_a_hit_sum = img_a_hit_sum_a - img_a_hit_sum_b

        
        img_a_hit_std = torch.transpose(img_a_hit_std, 1, 2)
        img_a_hit_sum = torch.transpose(img_a_hit_sum, 1, 2)

        
        img_a_hit_std = F.leaky_relu(self.Embed1a(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1b(img_a_hit_std))
        img_a_hit_std = F.leaky_relu(self.Embed1c(img_a_hit_std))
        
        img_a_hit_sum = F.leaky_relu(self.Embed2a(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2b(img_a_hit_sum))
        img_a_hit_sum = F.leaky_relu(self.Embed2c(img_a_hit_sum))

        img_a_hit_std_std  =  torch.std(img_a_hit_std, 2)
        img_a_hit_std_mean = torch.mean(img_a_hit_std, 2)

        img_a_hit_sum_std  =  torch.std(img_a_hit_sum, 2)
        img_a_hit_sum_mean = torch.mean(img_a_hit_sum, 2)
        
        xa1 = torch.cat((img_a_hit_std_std, img_a_hit_std_mean, img_a_hit_sum_std, img_a_hit_sum_mean), 1)
        xa1 = F.leaky_relu(self.embed_lin(xa1), 0.2)
        
        
        img_log_a_hit_std = torch.std(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_std_a = img_log_a_hit_std.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_std_b = img_log_a_hit_std.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_std = img_log_a_hit_std_a - img_log_a_hit_std_b
        
        img_log_a_hit_sum = torch.sum(img_log_a.view(batch_size, -1), 1)
        img_log_a_hit_sum_a = img_log_a_hit_sum.view(1, batch_size, 1).repeat(batch_size, 1, 1)
        img_log_a_hit_sum_b = img_log_a_hit_sum.view(batch_size, 1, 1).repeat(1, batch_size, 1)
        img_log_a_hit_sum = img_log_a_hit_sum_a - img_log_a_hit_sum_b
        
        img_log_a_hit_std = torch.transpose(img_log_a_hit_std, 1, 2)
        img_log_a_hit_sum = torch.transpose(img_log_a_hit_sum, 1, 2)
        
        img_log_a_hit_std = F.leaky_relu(self.Embed1a(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1b(img_log_a_hit_std))
        img_log_a_hit_std = F.leaky_relu(self.Embed1c(img_log_a_hit_std))
        
        img_log_a_hit_sum = F.leaky_relu(self.Embed2a(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2b(img_log_a_hit_sum))
        img_log_a_hit_sum = F.leaky_relu(self.Embed2c(img_log_a_hit_sum))

        img_log_a_hit_std_std  =  torch.std(img_log_a_hit_std, 2)
        img_log_a_hit_std_mean = torch.mean(img_log_a_hit_std, 2)

        img_log_a_hit_sum_std  =  torch.std(img_log_a_hit_sum, 2)
        img_log_a_hit_sum_mean = torch.mean(img_log_a_hit_sum, 2)        
        
        xa2 = torch.cat((img_log_a_hit_std_std, img_log_a_hit_std_mean, img_log_a_hit_sum_std, img_log_a_hit_sum_mean), 1)
        xa2 = F.leaky_relu(self.embed_lin(xa2), 0.2)
        
               
        xa3 = F.leaky_relu(self.diff_lin((img_a-img_b).view(batch_size, -1)), 0.2)
        
        
        xa4 = F.leaky_relu(self.diff_log((img_log_a-img_log_b).view(batch_size, -1)), 0.2)
 

        xa5 = F.leaky_relu(self.ln1a(self.conv1a(img_a*0.001)), 0.2)
        xa5 = F.leaky_relu(self.ln1b(self.conv1b(xa5)), 0.2)
        xa5 = F.leaky_relu(self.conv1c(xa5), 0.2)
        #print(xa5.size())
        xa5 = xa5.view(-1, self.ndf*6*6*6)
        xa5 = F.leaky_relu(self.conv_lin(xa5), 0.2)
        

        xa6 = F.leaky_relu(self.ln2a(self.conv2a(img_log_a)), 0.2)
        xa6 = F.leaky_relu(self.ln2b(self.conv2b(xa6)), 0.2)
        xa6 = F.leaky_relu(self.conv2c(xa6), 0.2)
        xa6 = xa6.view(-1, self.ndf*6*6*6)
        xa6 = F.leaky_relu(self.conv_log(xa6), 0.2)        
        
        xa7 = F.leaky_relu(self.econd_lin(E_true), 0.2)      
        
        xa = torch.cat((xa1, xa2, xa3, xa4, xa5, xa6, xa7), 1)
        
        
        xa = F.leaky_relu(self.fc1(xa), 0.2)
        xa = F.leaky_relu(self.fc2(xa), 0.2)
        xa = self.fc3(xa)

        
        return xa ### flattens
    
    
    

    
    
class BiBAE_F_3D_BatchStat_Core25(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_BatchStat_Core25, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(2,2,2), stride=(2,1,1),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([24,24,24])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(2,2,2), stride=(2,2,2),
                               padding=(0,0,0), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([12,12,12])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([6,6,6])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=(4,4,4), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*4+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z_full), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 7*7*12*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([24,14,14])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([48,28,28])

        self.conv0a = torch.nn.Conv3d(ngf, ngf, kernel_size=(1,2,2), stride=(1,1,1), padding=(0,0,0), bias=False)
        self.bnco0a = torch.nn.LayerNorm([48,27,27])
        self.conv0b = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,0,0), bias=False)
        self.bnco0b = torch.nn.LayerNorm([48,25,25])
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco0 = torch.nn.LayerNorm([48,25,25])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([48,25,25])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([48,25,25])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([48,25,25])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,48,25,25))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = x.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = x.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1,self.ngf,12,7,7)        

        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 24, 14, 14])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 48, 28, 28])), 0.2, inplace=True) #


        x = F.leaky_relu(self.bnco0a(self.conv0a(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco0b(self.conv0b(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        elif mode == 'half':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1))
            
    
    

        
        
class PostProcess_Size1Conv_EcondV2_Core25_Layers10_ndf128(nn.Module):
    def __init__(self, isize=48, isize2=25, nc=2, ndf=128, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_EcondV2_Core25_Layers10_ndf128, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.isize2 = isize2
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco5 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv6 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco6 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv7 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco7 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv8 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco8 = torch.nn.LayerNorm([self.isize, self.isize2, self.isize2])

        self.conv9 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv10 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize2, self.isize2) 
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize2*self.isize2*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize2, self.isize2)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.bnco5(self.conv5(img)), 0.01)
        img = F.leaky_relu(self.bnco6(self.conv6(img)), 0.01)
        img = F.leaky_relu(self.bnco7(self.conv7(img)), 0.01)
        img = F.leaky_relu(self.bnco8(self.conv8(img)), 0.01)
        img = F.leaky_relu(self.conv9(img), 0.01) 
        img = self.conv10(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize2, self.isize2)


     
