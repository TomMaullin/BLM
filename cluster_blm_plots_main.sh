rm log/*

j=1
while [ "$j" -le 15 ]; do
  i=36
  while [ "$i" -ge 26 ]; do
    echo $j
    qsub -N time_mem$i -V ./cluster_blm_plots.sh $(( $i )) $j
    i=$(($i - 1))
    qstat
    sleep 10
  done
  j=$(($j + 1))
  sleep $(( $j * 40 ))
done
