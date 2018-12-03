i=0
while [ $i-lt 10000 ]
do
	qstat
	sleep 1
	i=$(($i + 1))
done