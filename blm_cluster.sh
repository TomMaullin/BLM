#!/bin/bash

# include parse_yaml function
. lib/parse_yaml.sh

# Work out if we have been given multiple analyses configurations
# Else just assume blm_config is the correct configuration
if [ "$1" == "" ] ; then
  cfg=$(realpath "blm_config.yml")
else
  cfg=$1
fi

cfg=$(realpath $cfg)
# read yaml file to get output directory
eval $(parse_yaml $cfg "config_")
mkdir -p $config_outdir

# This file is used to record number of batches
if [ -f $config_outdir/nb.txt ] ; then
    rm $config_outdir/nb.txt 
fi
touch $config_outdir/nb.txt 

# Make a copy of the inputs file, if the user touches the cfg file this will not mess
# with anything now.
inputs=$config_outdir/inputs.yml
cp $cfg $inputs

fsl_sub -l log/ -N setup bash ./lib/cluster_blm_setup.sh $inputs > /tmp/$$ && setupID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$setupID" == "" ] ; then
  echo "Setup job submission failed!"
fi

echo "Setting up distributed analysis..."

# This loop waits for the setup job to finish before
# deciding how many batches to run. It also checks to 
# see if the setup job has errored.
nb=0
i=0
while [ $nb -lt 1 ]
do

  # obtain number of batches
  sleep 1
  if [ -s $config_outdir/nb.txt ]; then
    typeset -i nb=$(cat $config_outdir/nb.txt)
  fi
  i=$(($i + 1))

  # Check for error
  if [ $i -gt 30 ]; then
    errorlog="log/setup.e$setupID"
    if [ -s $errorlog ]; then
      echo "Setup has errored"
      exit
    fi
  fi

  # Timeout
  if [ $i -gt 500 ]; then
    echo "Something seems to be taking a while. Please check for errors."
  fi
done

echo "Submitting batch jobs..."

i=1
while [ "$i" -le "$nb" ]; do

  # Submit nb batches and get the ids for them
  fsl_sub -j $setupID -l log/ -N batch${i} bash ./lib/cluster_blm_batch.sh $i $inputs > /tmp/$$ && batchIDs=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$),$batchIDs

  i=$(($i + 1))
done
if [ "$batchIDs" == "" ] ; then
  echo "Batch jobs submission failed!"
fi

# Submit results job 
fsl_sub -j $batchIDs -l log/ -N results bash ./lib/cluster_blm_concat.sh $inputs > /tmp/$$ && resultsID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$resultsID" == "" ] ; then
  echo "Results job submission failed!"
fi
batchIDs=''

echo "Submitting results job..."
echo "Please use qstat to monitor progress."
