#!/bin/bash

python /home/ilc/rancherjobs/hadronGen/createMCParticle.py --pdg 22 --mass 0 --charge 0 --angleMax 40 --angleMin 40 --Emax 20 --Emin 20 \
| grep slcio  > /mnt/logFile.txt

LCIO_FILE=$(cat /mnt/logFile.txt)
mv $LCIO_FILE /mnt

exit 0;