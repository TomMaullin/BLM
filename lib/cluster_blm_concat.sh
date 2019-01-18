#!/bin/bash
#$ -S /bin/bash
#$ -q short.qc
#$ -cwd

module add fsl/5.0.11

fslpython -c "from BLM import blm_concat; blm_concat.main()"
