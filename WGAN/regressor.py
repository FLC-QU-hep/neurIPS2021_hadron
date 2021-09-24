from comet_ml import Experiment
import torch
import os, sys
sys.path.append(os.getcwd())
import functools
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import pdb
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import grad
import torch.nn.init as init


import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.spatial.distance as dist

## get/import models
from models.constrainer import *
from models.dataUtils import PionsDataset


def train(defparams, hyper):
    
    params = {}
    for param in defparams.keys():
        params[param] = defparams[param]

    hyperp = {}
    for hp in hyper.keys():
        hyperp[hp] = hyper[hp]

    experiment = Experiment(api_key="XXXXXXXXXXXXXX",
                        project_name="hadron-shower", workspace="engineren", auto_output_logging="simple")
    experiment.add_tag(params['exp'])

    experiment.log_parameters(hyperp)


    device = torch.device("cuda")       
    torch.manual_seed(params["seed"])

       
    aE = energyRegressor().to(device)
    
    optimizer_e = torch.optim.Adam(aE.parameters(), lr=hyperp["L_calib"], betas=(0.5, 0.9))
    

    aE = nn.parallel.DataParallel(aE, device_ids=[0])
 


    experiment.set_model_graph(str(aE), overwrite=False)
    

    if params["restore"]: 
        checkpoint = torch.load(params["restore_path"])
        aE.load_state_dict(checkpoint['Calib'])
        optimizer_e.load_state_dict(checkpoint['E_optimizer'])
        itr = checkpoint['iteration']
    
    else:
        aE.apply(weights_init)
        itr = 0
    
    
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)



    print("Loading data.................")
    paths_list = [
        '/path-to-data/.hdf5'
    ]


    train_data = PionsDataset(paths_list, core=True)
    

    dataloader = DataLoader(train_data, batch_size=hyperp["batch_size"], drop_last=True, num_workers=8, shuffle=True)
    

    print("Done --> loading data")



    e_criterion = nn.L1Loss() # for energy regressor training

    dataiter = iter(dataloader)
    
    BATCH_SIZE = hyperp["batch_size"]
    EXP = params["exp"]

    ## IO paths
    OUTP = params['output_path']


    for iteration in range(200000):
        
        iteration += itr + 1
        if iteration % 500 == 0.0:
            print("# iteration: ", iteration)

    
        batch = next(dataiter, None)

        if batch is None:
            dataiter = iter(dataloader)
            batch = dataiter.next()

        real_label = batch['energy'] ## energy label
        real_label = real_label.to(device)
    
        real_data = batch['shower'] 
        real_data = real_data.squeeze(1) #48x25x25 calo image
        real_data = real_data.to(device)

        #### supervised-training for energy regressor!
       
        
        output = aE(real_data.float())
        e_loss = e_criterion(output, real_label.view(BATCH_SIZE, 1))
        e_loss.backward()
        optimizer_e.step()
        
               
        experiment.log_metric("L_constrainer", e_loss, step=iteration)
        

        if iteration % 500 == 0 or iteration == 1 :

            torch.save({
            'Calib': aE.state_dict(),
            'E_optimizer': optimizer_e.state_dict(),
            'iteration': iteration
            },
            OUTP+'{0}/calib_itrs_{1}.pth'.format(EXP, iteration))
         
                
          
def main():
    
    default_params = {

        ## IO parameters
        "output_path" : '/your-disk-/output/',
        "exp" : 'wGANcore25x25_regressorLinp2',                   ## where the models will be saved!
        ## optimizer parameters 
        ## checkpoint parameters
        "restore" : True,
        "restore_path" : '/your-check-point/calib_itrs_61000.pth',
        "multi_gpu":False,
        "seed": 32,


    }

    hyper_params = {
        ## general 
        "batch_size" : 100,
        ## learning rate 
        "L_calib" : 1e-04

    }

    train(default_params,hyper_params)
    

if __name__ == "__main__":
    main()


    

    
    

    















