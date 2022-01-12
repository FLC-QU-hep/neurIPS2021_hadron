#!/bin/bash

source /home/ilc/ilcsoft/v02-01-pre/init_ilcsoft.sh

export REC_MODEL=ILD_l5_o1_v02

git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
cp ../* ./ILDConfig/StandardConfig/production
cd ./ILDConfig/StandardConfig/production
mv ./AHcalDigi.xml ./CaloDigi

run=$(echo $1 | cut -d'/' -f3 )

echo "-- Running Reconstruction--"

Marlin MarlinStdReco.xml --constant.lcgeo_DIR=$lcgeo_DIR \
        --constant.DetectorModel=${REC_MODEL} \
        --constant.OutputBaseName=pion_shower_$run \
        --constant.RunBeamCalReco=false \
        --global.LCIOInputFiles=$1


mv pion_shower_$run_REC.slcio /mnt/$run
