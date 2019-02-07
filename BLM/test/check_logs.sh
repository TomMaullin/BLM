# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Run each test case
i=1
for cfgIDs in $(ls ./BLM/test/cfgids/testIDs*)
do
  echo "Error logs for testcase $cfgIDs".
  cfgIDs=$(realpath $cfgIDs)

  # Read IDs from each line of file
  while read LINE; do 
    IDs=$($IDs $LINE); 
  done < $cfgIDs

  # Status update
  echo $IDs
done
