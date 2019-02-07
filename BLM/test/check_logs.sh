# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Run each test case
i=1
for cfgIDs in $(ls ./BLM/test/cfgids/testIDs*)
do
  echo "Error logs for testcase $cfgIDs".
  cfgIDs=$(realpath $cfgIDs)

  IDs=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH);exit}' $cfgIDs)

  # Status update
  echo $IDs
done
