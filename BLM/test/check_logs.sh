# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Run each test case
i=1
for cfgIDs in $(ls ./BLM/test/cfgIDs/)
do
  echo "Error logs for testcase $cfg".
  cfgIDs=$(realpath $cfgIDs)

  IDs=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH);exit}' )

  # Status update
  echo $IDs
done
