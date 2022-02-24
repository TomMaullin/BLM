#!/bin/bash

qsub -V -j $batchIDs -N results -e log/ -o log/ bash /well/nichols/users/inf852/BLMdask/scripts/cluster_blm_concat.sh 2627 /well/nichols/users/inf852/BLMdask/anya.yml