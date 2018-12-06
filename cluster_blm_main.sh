rm log/*

# include parse_yaml function
. parse_yaml.sh

# read yaml file to get output directory
eval $(parse_yaml BLM/blm_defaults.yml "config_")
echo $config_outdir

# This file is used to record number of batches
touch $config_outdir/nb.txt 

qsub -N setup -V cluster_blm_setup.sh

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
    errorlog=$(ls log/setup.e* | head -1)
    if [ -s $errorlog ]; then
      echo "Setup has errored"
      exit
    fi
  fi
done

echo "Submitting analysis jobs..."
i=1
while [ "$i" -le "$nb" ]; do
  qsub -N batch$i -V -hold_jid setup cluster_blm_batch.sh $i
  i=$(($i + 1))
done

qsub -N results -V -hold_jid "batch*" cluster_blm_concat.sh 
echo "Please use qstat to monitor progress."
