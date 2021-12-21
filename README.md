# Generative Models for Hadron Showers
We are motivated by the computational limitations of simulating interactions of particles in highly-granular detectors and trying to build fast and
exact machine-learning-based shower simulators. We present here the previously investigated WGAN and BIB-AE generative
models are improved and successful learning of hadronic showers initiated by charged
pions in a segment of the hadronic calorimeter of the International Large Detector
(ILD) is demonstrated for the first time 

This repository contains ingredients for repoducing *Hadrons, Better, Faster, Stronger* [[`arXiv:2112.09709`](https://arxiv.org/abs/2112.09709)]. A small fraction of the training data is available in [`Zenodo`](https://zenodo.org/record/5529677#.YcHUm9so-EI).

## Outline:

* [Generation of Tranining Data](#Generation-of-Training-Data)
* [Architectures](#Architectures)
* [Training](#Training)



## Generation of Tranining Data 

We use [`iLCsoft`](https://github.com/iLCSoft) ecosystem which includes `ddsim` and `Geant4`. 

### Kubernetes 
Assuming a running kubernetes cluster with volume attached, we can clone `ILDConfig` repository. 

Let's go to a `pod` which has our volume attached

```bash
rancher kubectl exec -it -n ilc ilcsoft-dev-c48b8775b-wjdm5 -- bash
cd /path/to/volume/attached/
```
Now we can clone the `ILDConfig` repository

```bash
git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
cd ILDConfig/StandardConfig/production
```


copy all `.py`, `.sh` , `create_root_tree.xml` and `pionGun.mac` files to this folder from `training_data` folder. 


## Running BIBAE code
Pytorch version 1.8.0

BIBAE_Train.py: Runfile for main BIBAE training

BIBAE_PP_Train.py: Runfile for main PostProcess training

## Running WGAN code
Pytorch version 1.8.0

wGAN.py: Main training file for WGAN

regressor.py: Training file for energy-regressor 
