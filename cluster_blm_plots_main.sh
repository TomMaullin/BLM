rm log/*

i=10
while [ "$i" -le 6063 ]; do
  qsub -N time$i -V ./cluster_blm_plots.sh $i
  i=$(($i + 1))
done