#!/bin/bash

qsub -V -N results -e /well/nichols/users/inf852/BLMdask/scripts/log/ -o /well/nichols/users/inf852/BLMdask/scripts/log/ bash /well/nichols/users/inf852/BLMdask/scripts/cluster_blm_concat.sh 2627 /well/nichols/users/inf852/BLMdask/anya.yml

fsl_sub -l log/ -N results bash /well/nichols/users/inf852/BLMdask/scripts/cluster_blm_concat.sh /well/nichols/users/inf852/BLMdask/scripts/cluster_blm_concat.sh 2627 /well/nichols/users/inf852/BLMdask/anya.yml