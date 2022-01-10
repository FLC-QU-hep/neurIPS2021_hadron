#!/bin/bash

source /home/ilc/ilcsoft/v02-01-pre/init_ilcsoft.sh

export REC_MODEL=ILD_l5_o1_v02

git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
cp ../* ./ILDConfig/StandardConfig/production
cd ./ILDConfig/StandardConfig/production
mv ./AHcalDigi.xml ./CaloDigi

echo "-- Running Reconstruction--"

Marlin MarlinStdReco.xml --constant.lcgeo_DIR=$lcgeo_DIR \
        --constant.DetectorModel=${REC_MODEL} \
        --constant.OutputBaseName=pion-shower \
        --constant.RunBeamCalReco=false \
        --global.LCIOInputFiles=/mnt/$1


mv pion-shower_REC.slcio /mnt 
