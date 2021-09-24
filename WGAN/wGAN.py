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


def getTotE(data, xbins, ybins, layers=48):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    etot_arr = np.sum(data, axis=(1))
    return etot_arr

def jsdHist(data_real, data_fake, nbins, minE, maxE):
    
    figSE = plt.figure(figsize=(6,6*0.77/0.67))
    axSE = figSE.add_subplot(1,1,1)

    
    pSEa = axSE.hist(data_real, bins=nbins, weights=np.ones_like(data_real)/(float(len(data_real))), range=[minE, maxE])
    pSEb = axSE.hist(data_fake, bins=nbins, weights=np.ones_like(data_fake)/(float(len(data_fake))), range=[minE, maxE])

    frq1 = pSEa[0]
    frq2 = pSEb[0]

    plt.close()
    # Jensen Shannon Divergence (JSD)
    score = dist.jensenshannon(frq1, frq2)
    print('Fidelity score: ', score)
    return score



def getHitE(data, xbins, ybins, layers):
    ehit_arr = np.reshape(data,[data.shape[0]*xbins*ybins*layers])
    #etot_arr = np.trim_zeros(etot_arr)
    ehit_arr = ehit_arr[ehit_arr != 0.0]
    return ehit_arr


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

    experiment = Experiment(api_key="keGmeIz4GfKlQZlOP6cit4QOi",
                        project_name="hadron-shower", workspace="engineren", auto_output_logging="simple")
    experiment.add_tag(params['exp'])

    experiment.log_parameters(hyperp)


    device = torch.device("cuda")       
    torch.manual_seed(params["seed"])

       
    
    aD = generate_model(hyperp["ndf"]).to(device)
    aG = DCGAN_G(hyperp["ngf"], hyperp["z"]).to(device)
    aE = energyRegressor().to(device)
    #aP = PostProcess_Size1Conv_EcondV2_CoreAnat(bias=True, out_funct='none').to(device)
    #aP = PostProcess_LinEcond_CoreAnat(bias=True, out_funct='none').to(device)

    
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=hyperp["L_gen"], betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=hyperp["L_crit"], betas=(0.5, 0.9))
    optimizer_e = torch.optim.SGD(aE.parameters(), lr=hyperp["L_calib"])
    #optimizer_p = torch.optim.Adam(aP.parameters(), lr=hyperp["L_post"], betas=(0.5, 0.9))



    if torch.cuda.device_count() > 1 :
    	aG = nn.parallel.DataParallel(aG)
    	aD = nn.parallel.DataParallel(aD)
    	aE = nn.parallel.DataParallel(aE)
    	#aP = nn.parallel.DataParallel(aP)

    else:
    	aG = nn.parallel.DataParallel(aG, device_ids=[0])
    	aD = nn.parallel.DataParallel(aD, device_ids=[0])
    	aE = nn.parallel.DataParallel(aE, device_ids=[0])
    	#aP = nn.parallel.DataParallel(aP, device_ids=[0])

       

    #experiment.set_model_graph(str(aG), overwrite=False)
    experiment.set_model_graph(str(aD), overwrite=False)

    
    if params["restore_pp"]:
        ppcheck = torch.load(params["restore_path_PP"])
        aP.load_state_dict(ppcheck['Postpro'])
        optimizer_p.load_state_dict(ppcheck['P_optimizer'])
        itr = ppcheck['iteration']
    #else:
    #    aP.apply(weights_init)

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

    if params["c0"]: 
        aE.apply(weights_init)
    
    elif params["c1"] :
        checkpoint = torch.load(params["calib_saved"])
        aE.load_state_dict(checkpoint['Calib'])
        optimizer_e.load_state_dict(checkpoint['E_optimizer'])
    
    
    
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)



    print("Loading data.................")
    paths_list = [
        '/beegfs/desy/user/eren/data_generator/pion/hcal_only/uniform/pion_uniform_510k_PunchThroughCut70.hdf5'
    ]


    train_data = PionsDataset(paths_list, core=True)
    

    dataloader = DataLoader(train_data, batch_size=hyperp["batch_size"], drop_last=True, num_workers=8, shuffle=True)
    

    dataiter = iter(dataloader)
    
    BATCH_SIZE = hyperp["batch_size"]
    LATENT = hyperp["z"]
    EXP = params["exp"]
    KAPPA = hyperp["kappa"]
    LAMBD = hyperp["lambda"]
    ## Post-Processing 
    LDP = hyperp["LDP"]
    wMMD = hyperp["wMMD"]
    wMSE = hyperp["wMSE"]

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

        if params["train_postP"]:
            #---------------------TRAIN P------------------------
            for p in aD.parameters():
                p.requires_grad_(False)  # freeze D
            
            for c in aG.parameters():
                c.requires_grad_(False)  # freeze G

            lossP = None
            for i in range(1):
                
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
                
                
                #Latent space Opt
                z_prime = lat_opt_ngd(aG, aD, noise, real_label, real_label.size()[0], device)
                ###

                z_prime.requires_grad_(True)

                real_data = batch['shower'] # 48x25x25 calo image
                real_data = real_data.to(device)

                fake_data = aG(z_prime, real_label.float())
                fake_data = fake_data.unsqueeze(1)  ## transform to [BS, 1, 48, 25, 25]

                
                ### first LossD_P
                fake_dataP = aP(fake_data.float(), real_label.view(BATCH_SIZE, 1))
                lossD_P = aD(fake_dataP.float(), real_label.view(BATCH_SIZE, 1))
                lossD_P = lossD_P.mean()

                ## lossFixP

                real_sorted = real_data.view(BATCH_SIZE, -1)
                fake_sorted = fake_dataP.view(BATCH_SIZE, -1)
                
                real_sorted, _ = torch.sort(real_sorted, dim=1, descending=True) 
                fake_sorted, _ = torch.sort(fake_sorted, dim=1, descending=True) #.view(900,1)

                lossFixPp1 = mmd_hit_sortKernel(real_sorted.float(), fake_sorted, kernel_size=50, stride=25, cutoff=500, alpha=100) 
                
                
                lossFixPp2 = F.mse_loss(fake_dataP.view(BATCH_SIZE, -1), 
                                        fake_data.detach().view(BATCH_SIZE, -1), reduction='mean')
                
                lossFixP = wMMD*lossFixPp1 + wMSE*lossFixPp2

                lossP = LDP*lossD_P - lossFixP

                lossP.backward(mone)            
                optimizer_p.step()

                experiment.log_metric("Post-processing MSE", lossFixPp2, step=iteration)
                experiment.log_metric("Post-processing MMD", lossFixPp1, step=iteration)

                ## monitor post-processing
                if iteration % 500 == 0.0:
                    #Generate 1k 50GeV showers 
                    noise_1k = np.random.uniform(-1, 1, (250, LATENT))
                    noise_1k = torch.from_numpy(noise_1k).float()
                    noise_1k = noise_1k.view(-1, LATENT, 1, 1, 1)    #[1000, nz]  --> [1000,nz,1,1,1] Needed for Generator
                    noise_1k = noise_1k.to(device)


                    tmp = np.full((250, 1), 50.0)
                    real_label_50GeV = torch.from_numpy(tmp).float()
                    real_label_50GeV = real_label_50GeV.view(-1, 1, 1, 1, 1)  #[1000,1] ---> [1000,1,1,1,1]  Needed for Generator
                    real_label_50GeV = real_label_50GeV.to(device)

                    z_prime_1k = lat_opt_ngd(aG, aD, noise_1k, real_label_50GeV, real_label_50GeV.size()[0], device)

                    fake_data_50GeV = aG(z_prime_1k, real_label_50GeV)
                    fake_data_50GeV_PP = aP(fake_data_50GeV.float(), real_label_50GeV.view(250, 1))

                    hit_fake = getHitE(fake_data_50GeV_PP.cpu().detach().numpy(), xbins=25, ybins=25, layers=48)
                    figHitE = plt.figure(figsize=(6,6))
                    axHitE = figHitE.add_subplot(1,1,1)
                    pHitEb = axHitE.hist(hit_fake, bins=np.logspace(np.log10(0.1),np.log10(10), 50), range=[0.1, 10], density=None, color='red',
                        histtype='stepfilled')
                    plt.yscale('log')
                    plt.xscale('log')
                    experiment.log_figure(figure=plt, figure_name="HitEnergy-50GeV")
                    del noise_1k, real_label_50GeV, z_prime_1k, fake_data_50GeV




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
            
            if params["train_postP"]:
                torch.save({
                    'Postpro': aP.state_dict(),
                    'P_optimizer': optimizer_p.state_dict(),
                    'iteration': iteration
                },
                OUTP+'{0}/netP_itrs_{1}.pth'.format(EXP, iteration))
    
                
          
                





        

def main():
    
    default_params = {

        ## IO parameters
        "output_path" : '/beegfs/desy/user/eren/HCAL-showers/WGAN/output/',
        "exp" : 'wGANcore25x25_rescritp13',                   ## where the models will be saved!
        "data_dim" : 3,
        ## optimizer parameters 
        "opt" : 'Adam',
        "gamma_g" : 1.0,                    ## not used at the moment 
        "gamma_crit" : 1.0,                 ## not used at the moment
        "gamma_calib" : 1.0,                ## not used at the moment
        ## checkpoint parameters
        "restore" : True,
        "restore_pp" : False,
        "restore_path" : '/beegfs/desy/user/eren/HCAL-showers/WGAN/output/wGANcore25x25_rescritp12/wgan_itrs_204000.pth',
        "restore_path_PP": '/beegfs/desy/user/eren/HCAL-showers/WGAN/output/wGANcoreLOUniPostPv2/netP_itrs_123000.pth',
        "calib_saved" : '/beegfs/desy/user/eren/HCAL-showers/WGAN/output/wGANcore25x25_regressorLinp2/calib_itrs_123000.pth',
        "post_saved" : 'netP_itrs_XXXX.pth', ##no need that actually
        "c0" : False,                   ## randomly starts calibration networks parameters
        "c1" : True,                    ## starts from a saved model
        "train_calib": False,           ## you might want to turn off constrainer network training
        "train_postP": False,
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
        "L_calib" : 1e-04,
        "L_post"  : 1e-04,
        ## model parameters
        "ngf" : 32,  
        "ndf" : 34,
        "z" : 100,
        ### hyper-parameters for post-processing
        "LDP" : 0.0,
        "wMMD" : 0.0,
        "wMSE" : 1.0,

    }

    train(default_params,hyper_params)
    

if __name__ == "__main__":
    main()


    

    
    

    















