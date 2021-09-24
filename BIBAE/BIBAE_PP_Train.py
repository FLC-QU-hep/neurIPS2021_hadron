import BAE_Pion_PP_Main
import time
import os

def main():
    # loss function weights for:     BIBAE   PostProc
    L_KLD,         L_P_KLD         = 0.1,    0.0      # Latent Regularization KLD 
    L_MMD_Latent,  L_P_MMD_Latent  = 100.0,  0.0      # Latent Regularization MMD
    L_MMD_HitKS,   L_P_MMD_HitKS   = 0.0,    5.0      # Sorted Kernel MMD for PostProc 1 
    L_MMD_HitKS2,  L_P_MMD_HitKS2  = 0.0,    5.0      # Sorted Kernel MMD for PostProc 2
    L_SortMSE,     L_P_SortMSE     = 0.0,    10.0     # Sorted Mean Squared Error for PostProc 
    L_SortMAE,     L_P_SortMAE     = 0.0,    10.0     # Sorted Mean Absolute Error for PostProc 
    L_batchcomp,   L_P_batchcomp   = 0.0,    0.0001   # Input/Output batch comparison for PostProc 


    empt = 0.0

    L_Losses =  [L_KLD,      L_MMD_Latent,   L_MMD_HitKS,   L_MMD_HitKS2,     
                 L_SortMAE,  L_SortMSE,      L_batchcomp]

    L_P_Losses =  [L_P_KLD,      L_P_MMD_Latent,   L_P_MMD_HitKS,   L_P_MMD_HitKS2,     
                   L_P_SortMAE,  L_P_SortMSE,      L_P_batchcomp]

    E_true_trans = lambda x:(x)  #100GeV -> 1.0

    params = {# loss functions
              'loss_List': L_Losses,
              "lossP_List" : L_P_Losses,
              'E_true_trans': E_true_trans,
        
        
              # model name and optional model suffix for saving
              'model': '3D_M_BiBAEBatchSCore_P_1ConvEcondV2L10H128_C_BatchSV2Core_CL_Default_Core25Thresh_PP',
              'suffix': '_test',
        
              
              # Data paths for I/O
              'input_path'  : '/beegfs/desy/user/diefenbs/shower_data/pion_uniform_510k_PunchThroughCut70.hdf5',
              'output_path' : '/beegfs/desy/user/diefenbs/VAE_results/test/',
              
              # Training settings. adapt to GPU setup's capabilities and available data
              'batch_size': 16,    # 
              'epochs': 100,
              'train_size': 96,   # adapt available data

              # Settings to resume training
              "continue_train": False,
              "continue_epoch": 0,

              # Settings for PostProcessor training without BIBAE trainings 
              "start_PostProc_after_ep":1,
              'BIBAE_Train' : False,
              
              # Learning rates for BIBAE, PostProcessor and the Critics
              'lr_VAE':1e-4*0.5,
              'lr_PostProc':1e-4*0.5,
              'lr_Critic':1e-4*0.5,
              'lr_Critic_E':0,
              'lr_Critic_L':2.0*1e-4,
              
              # Learning rate decay factors for BIBAE, PostProcessor and the Critics

              'gamma_VAE':0.97,
              "gamma_PostProc":0.97,
              'gamma_Critic':0.97,
              'gamma_Critic_E':0.97,
              'gamma_Critic_L':0.97}

    BAE_Pion_PP_Main.main(params)

if __name__ == '__main__':
    main()

