# Work out BLM dir
BLMdir=$(realpath ../../)
cd $BLMdir

# Read the test and data directories
testdir=$1
datadir=$2

# Change the name of the test and data directories in the test configurations
find ./BLM/test/cfg/fsltest_cfg*.yml -type f -exec sed -i 's/TEST_DIRECTORY/$testdir/g' {} \;
find ./BLM/test/cfg/fsltest_cfg*.yml -type f -exec sed -i 's/DATA_DIRECTORY/$datadir/g' {} \;

# Make a directory to store job ids if there isn't one already.
mkdir -p ./BLM/test/cfgids

# Run each test case
i=1
for cfg in $(ls ./BLM/test/cfg/fsltest_cfg*.yml)
do
  echo "Now running testcase $cfg".
  cfgfile=$(realpath $cfg)

  # Run blm for test configuration and save the ids
  bash ./blm_cluster.sh $cfgfile IDs > ./BLM/test/cfgids/fsltestIDs$(printf "%.2d" $i)tmp

  # Remove any commas from testIDs
  sed 's/,/ /g' ./BLM/test/cfgids/fsltestIDs$(printf "%.2d" $i)tmp > ./BLM/test/cfgids/fsltestIDs$(printf "%.2d" $i)
  rm ./BLM/test/cfgids/fsltestIDs$(printf "%.2d" $i)tmp

  # Status update
  qstat
  i=$(($i + 1))
done

# Include the parse yaml function
. lib/parse_yaml.sh

# Now run equivalent fsl analyses
for cfg in $(ls ./BLM/test/cfg/fsltest_cfg*.yml)
do
  # Obtain output directory
  cfgfile=$(realpath $cfg)
  eval $(parse_yaml $cfgfile "config_")

  echo $cfgfile
  echo $config_outdir
  echo $config_ns

  fsl_glm -i $datadir/$(config_ns)subNifti.nii.gz -d $datadir/X_$(config_ns)_fslformat.txt -o $(basename $cfg_outdir)/fsl/fsl_vox_betas -c $datadir/C_fslformat.txt --out_t=$(basename $cfg_outdir)/fsl/fsl_vox_Tstat_c --out_f=$(basename $cfg_outdir)/fsl/fsl_vox_Fstat_c --out_res=$(basename $cfg_outdir)/fsl/fsl_vox_resms --out_varcb=$(basename $cfg_outdir)/fsl/fsl_vox_cov_c

done
