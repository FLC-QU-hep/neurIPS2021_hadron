#!/bin/bash

source /home/ilc/ilcsoft/v02-01-pre/init_ilcsoft.sh

export SIM_MODEL=ILD_l5_v02

git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
cp ../* ./ILDConfig/StandardConfig/production
cd ./ILDConfig/StandardConfig/production

echo "-- Running DDSim $(SIM_MODEL) ..."
ddsim --outputFile ./pion-shower-{{ n }}.slcio \
             --compactFile ./compact/${SIM_MODEL}/${SIM_MODEL}.xml \
             --steeringFile ddsim_steer_gun.py 
