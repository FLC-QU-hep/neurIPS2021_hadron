from comet_ml import Experiment
import torch
import os, sys
sys.path.append(os.getcwd())
import functools
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
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
from models.generator import *
from models.constrainer import *
from models.dataUtils import PionsDataset
from models.criticRes import *



def calc_gradient_penalty(netD, real_data, fake_data, real_label, BATCH_SIZE, device, DIM):
    
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 1, 48, DIM, DIM)
    alpha = alpha.to(device)


    fake_data = fake_data.view(BATCH_SIZE, 1, 48, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)   

    disc_interpolates = netD(interpolates.float(), real_label.float())
    #disc_interpolates = netD(interpolates.float())

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def train(defparams, hyper):
    
    params = {}
    for param in defparams.keys():
        params[param] = defparams[param]

    hyperp = {}
    for hp in hyper.keys():
        hyperp[hp] = hyper[hp]

    experiment = Experiment(api_key="XXXXXXXXXXXXXXXX",
                        project_name="hadron-shower", workspace="engineren", auto_output_logging="simple")
    experiment.add_tag(params['exp'])

    experiment.log_parameters(hyperp)


    device = torch.device("cuda")       
    torch.manual_seed(params["seed"])

       
    
    aD = generate_model(hyperp["ndf"]).to(device)
    aG = DCGAN_G(hyperp["ngf"], hyperp["z"]).to(device)
    aE = energyRegressor().to(device)
   

    
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=hyperp["L_gen"], betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=hyperp["L_crit"], betas=(0.5, 0.9))
    
   



    if torch.cuda.device_count() > 1 :
    	aG = nn.parallel.DataParallel(aG)
    	aD = nn.parallel.DataParallel(aD)
    	aE = nn.parallel.DataParallel(aE)
    	

    else:
    	aG = nn.parallel.DataParallel(aG, device_ids=[0])
    	aD = nn.parallel.DataParallel(aD, device_ids=[0])
    	aE = nn.parallel.DataParallel(aE, device_ids=[0])
    	

       

    #experiment.set_model_graph(str(aG), overwrite=False)
    experiment.set_model_graph(str(aD), overwrite=False)

    
    if params["restore"]:   
        checkpoint = torch.load(params["restore_path"])
        aG.load_state_dict(checkpoint['Generator'])
        aD.load_state_dict(checkpoint['Critic'])
        optimizer_g.load_state_dict(checkpoint['G_optimizer'])
        optimizer_d.load_state_dict(checkpoint['D_optimizer'])
        itr = checkpoint['iteration']

    else:
        aG.apply(weights_init)
        itr = 0

    
    ## Load (or init) the constrainer-network
    if params["c0"]: 
        aE.apply(weights_init)
    
    elif params["c1"] :
        checkpoint = torch.load(params["calib_saved"])
        aE.load_state_dict(checkpoint['Calib'])
        
    
    
    
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)



    print("Loading data.................")
    paths_list = [
        '/path-to-data/.hdf5'
    ]


    train_data = PionsDataset(paths_list, core=True)
    

    dataloader = DataLoader(train_data, batch_size=hyperp["batch_size"], drop_last=True, num_workers=8, shuffle=True)
    

    dataiter = iter(dataloader)
    
    BATCH_SIZE = hyperp["batch_size"]
    LATENT = hyperp["z"]
    EXP = params["exp"]
    KAPPA = hyperp["kappa"]
    LAMBD = hyperp["lambda"]
    
   

    ## IO paths
    OUTP = params['output_path']


    for iteration in range(200000):
        
        iteration += itr + 1
        if iteration % 100 == 0.0:
            print("# iteration: ", iteration)

        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        

        ##starting with critic training
        aD.train()
        aG.eval()
        for i in range(hyperp["ncrit"]):
            
            #print("Critic training started")
            
            aD.zero_grad()
            
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, LATENT))    
            noise = torch.from_numpy(noise).float()
            noise = noise.view(-1, LATENT, 1, 1, 1)    #[BS, nz]  --> [Bs,nz,1,1,1] Needed for Generator
            noise = noise.to(device)
            
            batch = next(dataiter, None)

            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()

            real_label = batch['energy'] ## energy label
            real_label = real_label.to(device)
            

            with torch.no_grad():
                noisev = noise  # totally freeze G, training D

            fake_data = aG(noisev, real_label).detach()
            

            real_data = batch['shower'] # calo image
            real_data = real_data.to(device)
            real_data.requires_grad_(True)

        
            #### supervised-training for energy regressor!
            ## update: trained in a seperate script
            ######

            
            # train with real data
            disc_real = aD(real_data.float(), real_label.float())
        
            # train with fake data
            fake_data = fake_data.unsqueeze(1)  ## transform to [BS, 1, 48, 48, 48]
            disc_fake = aD(fake_data, real_label.float())

            
            # train with interpolated data
            gradient_penalty = calc_gradient_penalty(aD, real_data.float(), fake_data, real_label, BATCH_SIZE, device, DIM=25)
            
            ## wasserstein-1 distace
            w_dist = torch.mean(disc_fake) - torch.mean(disc_real)
            # final disc cost
            disc_cost = torch.mean(disc_fake) - torch.mean(disc_real) + LAMBD * gradient_penalty
            
            
            disc_cost.backward()
            optimizer_d.step()
            
            
            
            #--------------Log to COMET ML ----------
            if i == hyperp["ncrit"]-1:
                experiment.log_metric("L_crit", disc_cost, step=iteration)
                experiment.log_metric("gradient_pen", gradient_penalty, step=iteration)
                experiment.log_metric("Wasserstein Dist", w_dist, step=iteration)
                experiment.log_metric("Critic Score (Real)", torch.mean(disc_real), step=iteration)
                experiment.log_metric("Critic Score (Fake)", torch.mean(disc_fake), step=iteration)


        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D
        

        gen_cost = None
        ##Evaluate during generator training
        aD.eval()
        aG.train()
        aE.eval()
        for i in range(hyperp["ngen"]):
            
            aG.zero_grad()
            #print("Generator training started")
            
            
            noise = np.random.uniform(-1,1, (BATCH_SIZE, LATENT))
            noise = torch.from_numpy(noise).float()
            noise = noise.view(-1, LATENT, 1, 1, 1) #[BS, nz]  --> [Bs,nz,1,1,1] Needed for Generator
            noise = noise.to(device)


            batch = next(dataiter, None)

            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            
            real_label = batch['energy'] ## energy label
            real_label = real_label.to(device)
            
            
          
            noise.requires_grad_(True)

            real_data = batch['shower'] # 48x25x25 calo image
            real_data = real_data.to(device)


           
            fake_data = aG(noise, real_label.float())
            fake_data = fake_data.unsqueeze(1)  ## transform to [BS, 1, 48, 25, 25]
            
            
            ## calculate loss function 
            gen_cost = aD(fake_data.float(), real_label.float())
            
            ## label conditioning
            fake_data = fake_data.squeeze(1)  ## transform to [BS, 48, 25, 25]
            output_g = aE(fake_data)
            real_data = real_data.squeeze(1)  ## transform to [BS, 48, 25, 25]
            output_r = aE(real_data.float())


            aux_fake = (output_g - real_label)**2
            aux_real = (output_r - real_label)**2
            
            aux_errG = torch.abs(aux_fake - aux_real)
            
            ## Total loss function for generator
            g_cost = -torch.mean(gen_cost) + KAPPA*torch.mean(aux_errG) 
                
            
            g_cost.backward()
            optimizer_g.step()


            #--------------Log to COMET ML ----------
            experiment.log_metric("L_Gen_total", g_cost, step=iteration)
            experiment.log_metric("L_Gen", torch.mean(gen_cost), step=iteration)
            experiment.log_metric("Aux_errG * KAPPA", torch.mean(aux_errG)*KAPPA, step=iteration)
            

        

        #end = timer()
        #print(f'---train G elapsed time: {end - start}')

       

        if iteration % 500 == 0 or iteration == 1 :
            #print ('iteration: {}, critic loss: {}'.format(iteration, disc_cost.cpu().data.numpy()) )
            
            torch.save({
            'Generator': aG.state_dict(),
            'Critic': aD.state_dict(),
            'G_optimizer': optimizer_g.state_dict(),
            'D_optimizer': optimizer_d.state_dict(),
            'iteration': iteration
            },
            OUTP+'{0}/wgan_itrs_{1}.pth'.format(EXP, iteration))
            if params["train_calib"]:
                torch.save({
                'Calib': aE.state_dict(),
                'E_optimizer': optimizer_e.state_dict(),
                'iteration': iteration
                },
                OUTP+'{0}/calib_itrs_{1}.pth'.format(EXP, iteration))
            
            
    
                
          
                





        

def main():
    
    default_params = {

        ## IO parameters
        "output_path" : '/path-your-file-system/WGAN/output/',
        "exp" : 'wGANcore25x25_rescritp13',                   ## where the models will be saved!
        "data_dim" : 3,
        ## optimizer parameters 
        "opt" : 'Adam',
        ## checkpoint parameters
        "restore" : True,
        "restore_pp" : False,
        "restore_path" : '/path-to-your-checkpoint/wgan_itrs_204000.pth',
        "calib_saved" : '/path-to-your-constrainer/calib_itrs_123000.pth',
        "c0" : False,                   ## randomly starts calibration networks parameters
        "c1" : True,                    ## starts from a saved model
        "train_calib": False,           ## you might want to turn off constrainer network training
        "seed": 32,


    }

    hyper_params = {
        ## general 
        "batch_size" : 100,
        "lambda" : 25,
        "kappa" : 1.0,
        "ncrit" : 10,
        "ngen" : 1,
        ## learning rate 
        "L_gen" : 1e-05,
        "L_crit" : 1e-05,
        ## model parameters
        "ngf" : 32,  
        "ndf" : 34,
        "z" : 100,
        

    }

    train(default_params,hyper_params)
    

if __name__ == "__main__":
    main()


    

    
    

    















