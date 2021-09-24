import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data


class energyRegressor(nn.Module):
    """ 
    Energy regressor of WGAN. 
    """

    def __init__(self):
        super(energyRegressor, self).__init__()
        
        ## FC layers
        self.fc1 = torch.nn.Linear(48 * 25 * 25, 5000)
        self.bn1 = nn.BatchNorm1d(num_features=5000)
        self.drp1 = torch.nn.Dropout(p=0.2)
        
        self.fc2 = torch.nn.Linear(5000, 1000)
        self.bn2 = nn.BatchNorm1d(num_features=1000)
        self.drp2 = torch.nn.Dropout(p=0.1)
        
        self.fc3 = torch.nn.Linear(1000, 100)
        self.bn3 = nn.BatchNorm1d(num_features=100)
        self.fc4 = torch.nn.Linear(100, 1)
        
    def forward(self, x):
        #input shape :  [48, 25, 25]
        ## reshape the input: expand one dim
        #x = x.unsqueeze(1)
        
        ## image [48, 25, 25]
        ## flatten --> 30k cells 
        
        ## flatten for FC
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        
        ## pass to FC layers
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = self.drp1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.drp2(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.relu(self.fc4(x))
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
