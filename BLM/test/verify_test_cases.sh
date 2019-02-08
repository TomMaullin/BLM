# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Include the parse yaml function
. lib/parse_yaml.sh
cd $BLMdir/BLM/test

# Read the test directory
gtdir=$1
if [ -v $gtdir ] ; then
  echo "Please enter ground truth directory."
  exit
fi

# Run each test case
i=1
for cfg in $(ls ./cfg/test_cfg*.yml)
do
  
  cfgfilepath=$(realpath $cfg)

  echo " "
  echo " "
  echo "Now verifying: $(basename $cfgfilepath)".
  echo " "
  echo " "

  # read yaml file to get output directory
  eval $(parse_yaml $cfgfilepath "config_")
  
  fslpython -c "import verify_test_cases; verify_test_cases.main('$config_outdir', '$gtdir/$(basename $config_outdir)')"

  i=$(($i + 1))
done
