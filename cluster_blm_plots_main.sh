rm log/*

j=1
while [ "$j" -le 1 ]; do
  i=297
  while [ "$i" -le 297 ]; do
    echo $j
    qsub -N time$i -V ./cluster_blm_plots.sh $(( 10 * $i )) $j
    i=$(($i + 1))
  done
  j=$(($j + 1))
done
