rm log/*
rm BLM/binputs/*

# include parse_yaml function
. parse_yaml.sh

# read yaml file
eval $(parse_yaml BLM/blm_defaults.yml "config_")
echo $config_outdir

qsub -N setup -V cluster_blm_setup.sh

echo "Setting up distributed analysis..."

# This loop waits for the setup job to finish before
# deciding how many batches to run. It also checks to 
# see if the setup job has errored.
nb2=0
i=0
while [ $nb2 -lt 1 ]
do
  sleep 1
  if [ "$(ls -A BLM/binputs/)" ]; then
    sleep 3
    nb2=$(ls -1q BLM/binputs/Y* | wc -l)
  fi
  i=$(($i + 1))

  if [ $i -eq 10 ]; then
    echo "Verifying inputs..."
  fi

  if [ $i -eq 18 ]; then
    echo "Please wait..."
  fi

  if [ $i -gt 30 ]; then
    errorlog=$(ls log/setup.e* | head -1)
    if [ -s $errorlog ]; then
      echo "Setup has errored"
      exit
    fi
  fi
done

typeset -i nb = $(cat $(ls log/setup.o* | head -1))

i=1
while [ "$i" -le "$nb" ]; do
  qsub -N batch$i -V -hold_jid setup cluster_blm_batch.sh $i
  i=$(($i + 1))
done

qsub -N results -V -hold_jid "batch*" cluster_blm_concat.sh 
