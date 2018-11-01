#!/bin/bash
#$ -S /bin/bash
#$ -l h_rt=04:00:00
#$ -l h_vmem=16G
#$ -o log/
#$ -e log/
#$ -cwd

source /users/nichols/inf852/anaconda3/etc/profile.d/conda.sh

export PATH="/users/nichols/inf852/anaconda3/etc/profile.d/conda.sh$PATH"

conda activate DistOLSenv

python3 ./DistOLS-py/tmp_cluster.py

sleep 5m
