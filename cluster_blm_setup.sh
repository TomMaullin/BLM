#!/bin/bash
#$ -S /bin/bash
#$ -l h_rt=04:00:00
#$ -l h_vmem=16G
#$ -o log/
#$ -e log/
#$ -cwd

module add fsl/5.0.11

fslpython -c "from BLM import blm_setup; blm_setup.main()"
