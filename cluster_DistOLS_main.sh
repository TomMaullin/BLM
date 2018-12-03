rm log/*

qsub -N setup -V cluster_DistOLS_setup.sh

echo "Setting up distributed analysis..."

# This loop waits for the setup job to finish before
# deciding how many batches to run. It also checks to 
# see if the setup job has errored.
nb=0
i=0
while [ $nb -lt 1 ]
do
  echo $i
  sleep 1
  if [ "$(ls -A DistOLS/binputs/)" ]; then
    sleep 3
    nb=$(ls -1q DistOLS/binputs/Y* | wc -l)
  fi
  i=$(($i + 1))

  if [ $i -eq 10 ]; then
    echo "Verifying inputs..."
  fi

  if [ $i -eq 18 ]; then
    echo "Please wait..."
  fi

  if [ $i -gt 15 ]; then
    errorlog=$(ls log/setup.e* | head -1)
    if [ -s $errorlog ]; then
      echo "Setup has errored"
      exit
    fi
  fi
done

i=1
while [ "$i" -le "$nb" ]; do
  qsub -N batch$i -V -hold_jid setup cluster_DistOLS_batch.sh $i
  i=$(($i + 1))
done

qsub -N results -V -hold_jid "batch*" cluster_DistOLS_concat.sh 
