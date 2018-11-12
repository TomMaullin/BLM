rm DistOLS/binputs/*
rm log/*

qsub -N setup cluster_DistOLS_setup.sh

echo "Setting up distributed analysis..."

# This loop waits for the setup job to finish before
# deciding how many batches to run.
nb=0
while [ $nb -lt 1 ]
do
  sleep 1
  if [ "$(ls -A DistOLS/binputs/)" ]; then
    nb=$(ls -1q DistOLS/binputs/Y* | wc -l)
  fi
  errorlog=$(ls log/setup.e* | head -1)
  if [ -s $errorlog ]; then
    echo "Setup has errored"
    exit
  fi
done

i=1
while [ "$i" -le "$nb" ]; do
  qsub -N batch$i -hold_jid setup cluster_DistOLS_batch.sh $i
  i=$(($i + 1))
done

qsub -N results -hold_jid "batch*" cluster_DistOLS_concat.sh 
