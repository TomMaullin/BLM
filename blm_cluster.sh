#!/bin/bash

RealPath() {
    (echo $(cd $(dirname "$1") && pwd -P)/$(basename "$1"))
}

myqsub() {
    # Usage: myqsub "-N MyJob -q short.q" MyJob.sh arg1 arg2

    local Opts="$1"; shift

    # Adjust as per your local HPC configuration, so only thing output is task ID
    qsub -l log/ $Opts "$@" | awk '{print $3}'
    return ${PIPESTATUS[0]}
}

myqsubtask() {
    # Usage: myqsubtask "-N MyJob -q short.q" 1-10 MyJob.sh arg1 arg2
    # But *note* MyJob.sh must take an intial argument with the task array number:
    #     MyJob.sh TaskId arg1 arg2

    local Opts="$1"; shift
    local TaskIDrng="$2"; shift
    local Cmd="$2"; shift

    local TmpFile="$(mktemp -t blm_)"

    cat << EOF > $TmpFile
#!/bin/bash
$Cmd \$SGE_TASK_ID "$@"
EOF
    chmod +x $TmpFile
    # Adjust as per your local HPC configuration, so only thing output is task ID *only*
    qsub -l log/ $Opts -t "$TaskIDrng" $TmpFile | awk '{print $3}' | awk -F. '{print $1}'
    return ${PIPESTATUS[0]}
}



BLM_PATH=$(dirname $(RealPath "${BASH_SOURCE[0]}"))

# include parse_yaml function
. $BLM_PATH/scripts/parse_yaml.sh

# Work out if we have been given multiple analyses configurations
# Else just assume blm_config is the correct configuration
if [ "$1" == "" ] ; then
  cfg=$(RealPath "blm_config.yml")
else
  cfg=$1
fi

# If the second argument is IDs we use this to print IDs
if [ "$2" == "IDs" ] ; then
  printOpt=2
else
  printOpt=1
fi

cfg=$(RealPath $cfg)
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

myqsub "-N setup" bash $BLM_PATH/scripts/cluster_blm_setup.sh $inputs > /tmp/$$ && setupID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$setupID" == "" ] ; then
  echo "Setup job submission failed!"
fi

if [ "$printOpt" == "1" ] ; then
  echo "Setting up distributed analysis..."
else
  echo $setupID
fi

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

if [ "$printOpt" == "1" ] ; then
  echo "Submitting batch jobs..."
fi

# Reread yaml file in case filepaths have been updated to be absolute
eval $(parse_yaml $inputs "config_")
inputs=$config_outdir/inputs.yml

i=1
while [ "$i" -le "$nb" ]; do

  # Submit nb batches and get the ids for them
  myqsub "-j $setupID -N batch${i}" bash $BLM_PATH/scripts/cluster_blm_batch.sh $i $inputs > /tmp/$$ && batchIDs=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$),$batchIDs
  i=$(($i + 1))
done
if [ "$batchIDs" == "" ] ; then
  echo "Batch jobs submission failed!"
else
  if [ "$printOpt" == "2" ] ; then
    echo $batchIDs
  fi
fi

# Submit results job 
myqsub "-j $batchIDs -N results" bash $BLM_PATH/scripts/cluster_blm_concat.sh $inputs > /tmp/$$ && resultsID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$resultsID" == "" ] ; then
  echo "Results job submission failed!"
fi

if [ "$printOpt" == "1" ] ; then
  echo "Submitting results job..."
else
  echo $resultsID
fi

# -----------------------------------------------------------------------
# Submit Cleanup job
# -----------------------------------------------------------------------
myqsub "-j $resultsID -N cleanup" bash $BLM_PATH/scripts/cluster_blm_cleanup.sh $inputs > /tmp/$$ && cleanupID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$cleanupID" == "" ] ; then
  echo "Clean up job submission failed!"
fi

if [ "$printOpt" == "1" ] ; then
  echo "Submitting clean up job..."
  echo "Analysis submission complete. Please use qstat to monitor progress."
else
  echo $cleanupID
fi

