rm log/*

i=22
while [ "$i" -le 52 ]; do
  qsub -N time$i -V ./cluster_blm_plots.sh $(( $i ))
  i=$(($i + 1))
  qstat
done
