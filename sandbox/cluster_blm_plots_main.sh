rm log/*

j=8
while [ "$j" -le 10 ]; do
  i=10
  while [ "$i" -le 15 ]; do
    echo $j
    qsub -N time_mem$i -V ./cluster_blm_plots.sh $(( $i )) $j
    i=$(($i + 1))
    qstat
  done
  j=$(($j + 1))
  sleep 1800
done
