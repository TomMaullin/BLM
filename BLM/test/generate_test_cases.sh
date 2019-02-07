# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Read the test and data directories
testdir=$1
datadir=$2

# Change the name of the test and data directories in the test configurations
find ./BLM/test/cfg/*.yml -type f -exec sed -i 's/TEST_DIRECTORY/$testdir/g' {} \;
find ./BLM/test/cfg/*.yml -type f -exec sed -i 's/DATA_DIRECTORY/$datadir/g' {} \;

# Make a directory to store job ids if there isn't one already.
mkdir -p ./BLM/test/cfgids

# Run each test case
i=1
for cfg in $(ls ./BLM/test/cfg/*.yml)
do
  echo "Now running testcase $cfg".
  cfgfile=$(realpath $cfg)

  # Run blm for test configuration and save the ids
  bash ./blm_cluster.sh $cfgfile > ./BLM/test/cfgids/testIDs$i

  # Status update
  qstat
  i=$(($i + 1))
done
