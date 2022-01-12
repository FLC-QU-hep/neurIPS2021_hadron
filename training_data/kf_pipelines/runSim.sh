#!/bin/bash

source /home/ilc/ilcsoft/v02-01-pre/init_ilcsoft.sh

export SIM_MODEL=ILD_l5_v02

git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
cp ../* ./ILDConfig/StandardConfig/production
cd ./ILDConfig/StandardConfig/production

n=$(echo $RANDOM)

echo "-- Running DDSim..."
ddsim --outputFile ./pion-shower_$n.slcio --compactFile $lcgeo_DIR/ILD/compact/${SIM_MODEL}/${SIM_MODEL}.xml --steeringFile ddsim_steer_gun.py 

echo "Converting: LCIO --> root file"
Marlin create_root_tree.xml --global.LCIOInputFiles=./pion-shower_$n.slcio --MyAIDAProcessor.FileName=pion-shower_$n;

mv ./pion-shower_$n.slcio /mnt
mv ./pion-shower_$n.root /mnt

echo pion-shower_$n.slcio > /mnt/simout
echo pion-shower_$n.root > /mnt/simout_root