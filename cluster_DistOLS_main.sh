qsub -N setup cluster_DistOLS_setup.sh

nb=$(ls -1q DistOLS/binputs/Y* | wc -l)

i=1
while [ "$i" -le "$nb" ]; do
  qsub -N batch$i -hold_jid setup cluster_DistOLS_batch.sh $i
  i=$(($i + 1))
done

#qsub -N results -hold_jid "batch*" cluster_DistOLS_results.sh 
