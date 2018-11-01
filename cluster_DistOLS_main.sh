qsub -N setup cluster_DistOLS_setup.sh

for i in {1..10}
do
	qsub -N batch$i -hold_jid setup cluster_DistOLS_batch.sh
done

qsub -N results -hold_jid "batch*" cluster_DistOLS_results.sh 
