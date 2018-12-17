#!/bin/bash
#$ -S /bin/bash
#$ -q short.qc
#$ -o log/
#$ -e log/
#$ -cwd


module add fsl/5.0.11

fslpython -c "import local_blm_plots; local_blm_plots.main($1)"
