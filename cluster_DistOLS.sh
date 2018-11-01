#!/bin/bash
#$ -S /bin/bash
#$ -l h_rt=04:00:00
#$ -l h_vmem=16G
#$ -t 1:38
#$ -o log/
#$ -e log/
#$ -cwd

source activate DistOLSenv

python3 tmp_cluster.py