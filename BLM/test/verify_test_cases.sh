# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Read the test directory
gtdir=$1

# Run each test case
i=1
for cfg in $(ls ./BLM/test/cfg/test_cfg*.yml)
do
  
  cfgfilepath=$(realpath $cfg)
  cfgfolder=$(basename $cfg)

  echo "Now verifying testcase $cfgfolder".

  # read yaml file to get output directory
  eval $(parse_yaml $cfgfilepath "config_")
  fslpython -c "from verify_test_cases; verify_test_cases.main("$config_outdir", "$gtdir/$cfgfolder")"

  i=$(($i + 1))
done
