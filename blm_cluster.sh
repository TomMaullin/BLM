#!/bin/bash

# include parse_yaml function
. lib/parse_yaml.sh

# read yaml file to get output directory
eval $(parse_yaml blm_config.yml "config_")
mkdir -p $config_outdir

# This file is used to record number of batches
if [ -f $config_outdir/nb.txt ] ; then
    rm $config_outdir/nb.txt 
fi
touch $config_outdir/nb.txt 

jID=`fsl_sub -l log/ -N setup bash ./lib/cluster_blm_setup.sh`
setupID=`echo $jID | awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}'`
qstat

echo $setupID

echo "Setting up distributed analysis..."

# This loop waits for the setup job to finish before
# deciding how many batches to run. It also checks to 
# see if the setup job has errored.
nb=0
i=0
while [ $nb -lt 1 ]
do
  sleep 1
  if [ -s $config_outdir/nb.txt ]; then
    typeset -i nb=$(cat $config_outdir/nb.txt)
  fi
  i=$(($i + 1))

  if [ $i -gt 30 ]; then
    errorlog="log/setup.e$setupID"
    if [ -s $errorlog ]; then
      echo "Setup has errored"
      exit
    fi
  fi

  if [ $i -gt 500 ]; then
    echo "Something seems to be taking a while. Please check for errors."
  fi
done

echo "Submitting analysis jobs..."
i=1
while [ "$i" -le "$nb" ]; do
  qsub -o log$1/ -e log$1/ -N batch$i -V -hold_jid setup lib/cluster_blm_batch.sh $i
  i=$(($i + 1))
done

qsub -o log$1/ -e log$1/ -N results -V -hold_jid "batch*" lib/cluster_blm_concat.sh
echo "Please use qstat to monitor progress."
