rm log/*

i=10
while [ "$i" -le 600 ]; do
  qsub -N time$i -V ./cluster_blm_plots.sh $(( 10 * $i ))
  i=$(($i + 1))
  qstat
  sleep $(( $i / 5 ))
  qstat
done
