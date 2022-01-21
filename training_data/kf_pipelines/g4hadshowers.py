#!/usr/bin/env python3
# Copyright 2019 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kfp
from kfp import dsl
from kfp.components import InputPath, InputTextFile, InputBinaryFile, OutputPath, OutputTextFile, OutputBinaryFile


def create_vol():
    return dsl.VolumeOp(
        name="persistent-volume",
        resource_name="my-pvc",
        modes=dsl.VOLUME_MODE_RWO,
        size="15Gi"
    )

def sim(v):
    return dsl.ContainerOp(
                    name='Simulation',
                    image='ilcsoft/ilcsoft-centos7-gcc8.2:v02-01-pre',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone --branch postpaper https://github.com/FLC-QU-hep/neurIPS2021_hadron.git  && \
                                cd $PWD/neurIPS2021_hadron/training_data/kf_pipelines/ && chmod +x ./runSim.sh && ./runSim.sh && \
                                ls -ltrh /mnt'],
                    pvolumes={"/mnt": v.volume},
                    file_outputs={'lcio': '/mnt/lcio_path',
                                   'root': '/mnt/root_path'
                    },
    )    

def rec(v, simout_name):
    return dsl.ContainerOp(
                    name='Reconstruction',
                    image='ilcsoft/ilcsoft-centos7-gcc8.2:v02-01-pre',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone --branch postpaper https://github.com/FLC-QU-hep/neurIPS2021_hadron.git && \
                                cd $PWD/neurIPS2021_hadron/training_data/kf_pipelines/ && \
                                chmod +x ./runRec.sh && ./runRec.sh "$0"', simout_name ],
                    pvolumes={"/mnt": v.volume}
    )   


def convert_hdf5(v, simout_root, file_name):
    return dsl.ContainerOp(
                    name='hdf5 conversion',
                    image='engineren/pytorch:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone --branch postpaper https://github.com/FLC-QU-hep/neurIPS2021_hadron.git && \
                                cd $PWD/neurIPS2021_hadron/training_data/kf_pipelines/ && cp ../create_hdf5.py . && \
                                python create_hdf5.py --rootfile "$0" --branch pionSIM --batchsize 1 --output pion-shower-"$1" --hcal True && \
                                RUN=$(echo "$0" | cut -d"/" -f3) && \
                                cp pion-shower-"$1".hdf5 /mnt/$RUN && cp pion-shower-"$1".hdf5 /tmp', simout_root, file_name],
                    pvolumes={"/mnt": v.volume},
                    file_outputs = {
                        'data': '/tmp/pion-shower-'+file_name+'.hdf5',
                    }
    )   


def train(v, hdf5_data):
    return dsl.ContainerOp(
                    name='train',
                    image='engineren/pytorch:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['echo "$0" && python -c "import torch; print(torch.__version__)" ', hdf5_data],
                    pvolumes={"/mnt": v.volume}
    )   




@dsl.pipeline(
    name='ILDEventGen',
    description='Event Simulation and Reconstruction'
)

def sequential_pipeline():
    """A pipeline with sequential steps."""
    
    r = create_vol()
    simulation = sim(r)
    recost = rec(r, simulation.outputs['lcio'])
    hdf5 = convert_hdf5(r, simulation.outputs['root'], "p1")
    
    training_data = dsl.InputArgumentPath(hdf5.output) 
    tr = train(r, training_data)
   

   
    
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, __file__ + '.yaml')
