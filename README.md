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

### Kubernetes (in our case via Rancher) 
Assuming a running Kubernetes cluster with volume attached, we clone `ILDConfig` repository. 

Let's go to a `pod` which has our volume attached to `/mnt` folder

```bash
engin@local: ~$ rancher kubectl exec -it -n ilc ilcsoft-dev-c48b8775b-wjdm5 -- bash
## you're inside the container in k8s
bash-4.2# cd /mnt
```
Now we can clone the `ILDConfig` repository (+ this repository!)

```bash
bash-4.2# git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
bash-4.2# cd ./ILDConfig/StandardConfig/production
bash-4.2# cp ./neurIPS2021_hadron/training_data/* .
```
For the sake of this example, now, we can start 10 simulation jobs each generating 100 events 

```bash
## go back to your local
engin@local: ~$ alias render_template='python -c "from jinja2 import Template; import sys; print(Template(sys.stdin.read()).render());"'
engin@local: ~$ cat sim.jinja2 | render_template > sim_jobs.yaml 
```
and start submitting

```console
engin@local: ~$ rancher kubectl apply -f sim_jobs.yaml -n ilc
job.batch/pion-sim-1 created
job.batch/pion-sim-2 created
job.batch/pion-sim-3 created
job.batch/pion-sim-4 created
job.batch/pion-sim-5 created
job.batch/pion-sim-6 created
job.batch/pion-sim-7 created
job.batch/pion-sim-8 created
job.batch/pion-sim-9 created
job.batch/pion-sim-10 created
```
These jobs will create Linear-Collider-Input-Output (LCIO) and root files, containing all information about the event in the realistic ILD simulation.

We need to do:

1. Read/stream the root files and convert all calorimeter hits into `48 x 48 x 48` numpy array.
2. Write these arrays into a `hdf5` file. 

We have a different workflow (with different docker image) for this stage:

```console
engin@local: ~$ cat convert_hdf5.jinja2 | render_template > convert_hdf5_jobs.yaml
engin@local: ~$ rancher kubectl apply -f convert_hdf5_jobs.yaml -n ilc
job.batch/hdf5-pions-1 created
job.batch/hdf5-pions-2 created
job.batch/hdf5-pions-3 created
job.batch/hdf5-pions-4 created
job.batch/hdf5-pions-5 created
job.batch/hdf5-pions-6 created
job.batch/hdf5-pions-7 created
job.batch/hdf5-pions-8 created
job.batch/hdf5-pions-9 created
job.batch/hdf5-pions-10 created
```

We further cut the size of transverse grid to `25x25` in order to avoid too low spacity. This stage is done in the data loader.  

### Kubeflow-Pipelines (Local)

Kubeflow Pipelines are powerful. We can orchestrate simple, portable and scalable ML workflows. However, its installation process might be painful and require extra help from IT guys. Here we will create a local cluster and install Kubeflow-Pipelines

Before you start, make sure you have:

1. Docker with ~10GB RAM reserved
2. Helm v3.6.3
3. kind

Create a local kind cluster
```bash
engin@local: ~$ kind create cluster --name kfp
```

it might take 1-2 minutes to spin up a local cluster.

Install Kubeflow Pipelines from [`GetInData's Helm Chart`](https://getindata.com/blog/kubeflow-pipelines-running-5-minutes/)

```bash
engin@local: ~$ helm repo add getindata https://getindata.github.io/helm-charts/
engin@local: ~$ helm install my-kubeflow-pipelines getindata/kubeflow-pipelines \
  --version 1.6.2 --set platform.managedStorage.enabled=false \
  --set platform.cloud=gcp  --set platform.gcp.proxyEnabled=false
```

Now be patient; it might take few minutes. Don't worry if you see `ml-pipeline` or `metadata-grpc-deployment` pods having a `CrashLoopBackOff` state for some time. They will become ready once their dependent services launch. 

```bash
engin@local: ~$ kubectl get pods
NAME                                              READY   STATUS      RESTARTS   AGE
cache-deployer-deployment-db7bbcff5-h6vl8         1/1     Running     12         11d
cache-server-748468bbc9-cd5ls                     1/1     Running     10         11d
metadata-envoy-7cd8b6db48-pdr27                   1/1     Running     10         11d
metadata-grpc-deployment-7c9f96c75-c2jsj          1/1     Running     21         11d
metadata-writer-78f67c4cf9-5wdcp                  1/1     Running     13         11d
minio-6d84d56659-dcdkr                            1/1     Running     10         11d
ml-pipeline-8588cf6787-ht226                      1/1     Running     30         11d
ml-pipeline-persistenceagent-b6f5ff9f5-pjlz9      1/1     Running     17         11d
ml-pipeline-scheduledworkflow-6854cdbb8d-cdd5q    1/1     Running     10         11d
ml-pipeline-ui-cd89c5577-jrrg9                    1/1     Running     10         11d
ml-pipeline-viewer-crd-6577dcfc8-zrvrg            1/1     Running     16         11d
ml-pipeline-visualizationserver-f9895dfcd-pkcr7   1/1     Running     10         11d
mysql-6989b8c6f6-mz2cl                            1/1     Running     10         11d
workflow-controller-6d457d9fcf-w74q8              1/1     Running     15         11d
```
Now we can go ahead and access UI: 

```bash
engin@local: ~$ kubectl port-forward svc/ml-pipeline-ui 9000:80
```
Open this browser: `http://localhost:9000/#/pipelines`

![UI][figures/kfp-UI.png]

In order to pause the cluster

```bash
engin@local: ~$ docker ps 
CONTAINER ID   IMAGE                  COMMAND                  CREATED       STATUS          PORTS                       NAMES
486ef2b14930   kindest/node:v1.21.1   "/usr/local/bin/entr…"   11 days ago   Up 31 minutes   127.0.0.1:36553->6443/tcp   kfp-control-plane
engin@local: ~$ docker stop 486ef2b14930
```
use `docker start CONTAINER ID` to start again. 



## Running BIBAE code
Pytorch version 1.8.0

BIBAE_Train.py: Runfile for main BIBAE training

BIBAE_PP_Train.py: Runfile for main PostProcess training

## Running WGAN code
Pytorch version 1.8.0

wGAN.py: Main training file for WGAN

regressor.py: Training file for energy-regressor 
