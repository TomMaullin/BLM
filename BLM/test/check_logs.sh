# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Run each test case
i=1
for cfgIDs in $(ls ./BLM/test/cfgids/*)
do
  echo " "
  echo '================================================================'
  echo "Error logs for testcase $(basename $cfgIDs)"
  echo " "
  cfgIDs=$(realpath $cfgIDs)

  # Read IDs from each line of file
  while read LINE; do 
    IDs=$IDs" "$LINE; 
  done < $cfgIDs

  # Status update
  for ID in $IDs
  do
    logfile=$(ls ./log/*.e$ID | head -1)
    logfilecontent=$(cat $logfile)
    if [ ! -z "$logfilecontent" ] ; then
      echo "Logged error in: $logfile"
      echo $logfilecontent
      logsencountered=1
      echo " "
    fi
  done

  if [ -z "$logsencountered" ] ; then
    echo "No errors encountered for testcase $i"
  fi

  IDs=''
  i=$(($i + 1))
  logsencountered=""
done
