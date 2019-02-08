# BLM-py
This repository contains the code for Big Linear Models for Neuroimaging cluster and local usage.

## Requirements
To use the BLM-py code, `fsl 5.0.10` or greater must be installed and `fslpython` must be configured correctly. Alternatively the following python packages must be installed:

```
numpy>=1.14.0
nibabel>=2.2.1
yaml
pandas
subprocess
```

(This code may work with older versions of numpy and nibabel but caution is advised as these versions have not been tested).

If running `BLM-py` on a cluster, `fsl_sub` must also be configured correctly.

## Usage
To run `BLM-py` first specify your design using `blm_config.yml` and then run the job either in serial or in parallel by following the below guidelines.

### Specifying your model
The regression model for BLM must be specified in `blm_config.yml`. Below is a complete list of possible inputs to this file.

#### Mandatory fields
The following fields are mandatory:

 - `MAXMEM`: This is the maximum amount of memory (in bits) that the BLM code is allowed to work with. How this should be set depends on your machine capabilities but the recommended value matches the SPM default of 2^32 (note this must be in python notation i.e. `2**32`).
 - `Y_files`: This should be a text file containing a list of response variable NIFTI's.
 - `X`: This should be the design matrix as a plain csv (no column headers).
 - `outdir`: This is the output directory.
 - `contrasts`: These are the contrast vectors to be tested. They should be listed as `c1,c2,...` etc and each contrast should contain the fields:
   - `name`: A name for the contrast. i.e. `awesomelynamedcontrast1`.
   - `vector`: A vector for the contrast. i.e. `[1, 0, 0]` or `[[1, 0, 0],[0,1,0]]`
   - `statType`: The statistic type of the contrast vector (`T` or `F`).
   
   At least one contrast must be given, see `Examples` for an example of how to specify contrasts.
 
#### Optional fields

The following fields are optional:

 - `M_files`: A text file containing a list of masks to be applied to the `Y_files`. 
   - If the number of masks is the same as the number of `Y_files` then each mask is applied to the corresponding entry `Y_files`. E.g. The first mask listed for `M_files` will be applied to the first nifti in `Y_files`, the second mask in `M_files` will be applied to the second nifti in `Y_files` and so on. 
   - If the number of masks is less than the number of niftis in `Y_files` then every mask is applied to every nifti in `Y_files`. I.e. all masks are applied to the first nifti in `Y_files`, all masks are applied to the second nifti in `Y_files` and so on. 
 - `Missingness`: This field allows the user to mask the image based on how many studies had recorded values for each voxel. This can be specified in 3 ways.
   - `MinPercent`: The percentage of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `0.1` then any voxel with recorded values for at least 10% of studies will be kept in the analysis.
   - `MinN`: The number of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `20` then any voxel with recorded values for at least 20 studies will be kept in the analysis.
   - `Masking`: A post analysis mask.
 - `OutputCovB`: If set to `True` this will output between beta covariance maps. For studies with a large number of paramters this may not be desirable as, for example, 30 analysis paramters will create 30x30=900 between beta covariance maps. By default this is set to `True`.
 
#### Running an analysis in parallel

To run an analysis in parallel, log into the cluster you wish to run it on and ensure that `fsl` and `fsl_sub` are in the environment. On the `rescomp` cluster this can be done like so:

```
module add fsl
module add fsl_sub
```

Ensure you are in the `BLM-py` directory and once you are happy with the analysis you have specified in `blm_config.yml`, run the following command:

```
bash ./blm_cluster.sh
```

After running this you will see text printed to the commandline telling you the analysis is being set up and the jobs are being submitted. For large analyses or small cluster this may take a minute or two as several jobs may be submitted to the cluster. Once you can access the command line again, you can use `qstat` to see the jobs which have been submitted. You will typically see jobs of the following form:

 - `setup`: This will be working out the number of batches/blocks the analysis needs to be split into.
 - `batch*`: There may be several jobs with names of this format. These are the "chunks" the analysis has been split into. These are run in parallel to one another and typically don't take very long.
 - `results`: This code is combining the output of each batch to obtain statistical analyses. This will run once all `batch*` jobs have been completed. Please note this code has been streamlined for large numbers of subjects but not large number of parameters; therefore this job may take some time for large numbers of parameters.
 
#### Running an analysis in serial

To run an analysis in serial, ensure you are in the `BLM-py` directory and once you are happy with the analysis you have specified in `blm_config.yml`, run the following command:

```
fslpython -c "import blm_serial; blm_serial.main()
```

The commandline will then tell you how much progress is being made as it runs each block.

## Testing

Note: All the below tests require access to test data. To ask access, please ask @TomMaullin directly.

### In parallel, against ground truth

To generate test cases:

```
bash ./generate_test_cases.sh $outdir $datadir
```

(Where `$datadir` is a data directory containg all data needed for analyses `test_cfg01.yml`, `test_cfg02.yml`,... `test_cfg10.yml` and `$outdir` is the desired output directory)

To check the logs:

```
bash ./check_logs.sh
```

To verify the test cases against ground truth:

```
bash ./verify_test_cases.sh $GTDIR
```

(Where `$GTDIR` is a directory containing ground truth data from a previous run, i.e. inside `$GTDIR` are the folders `test_cfg01.yml`, `test_cfg02.yml`, ... ect).

### In parallel, against FSL

To generate test cases:

```
bash ./generate_test_cases_fsl.sh $outdir $datadir
```

(Where `$datadir` is a data directory containg all data needed for designs `fsltest_cfg01.yml`, `fsltest_cfg02.yml` and `fsltest_cfg03.yml` and `$outdir` is the desired output directory)

To check the logs:

```
bash ./check_logs.sh
```

To verify the test cases against ground truth:

```
bash ./verify_test_cases_against_fsl.sh
```

### In serial

To test in serial simply run the following 3 test cases from the main `BLM-py` folder:

```
fslpython -c "import blm_serial; blm_serial.main('./BLM/test/cfg/test_cfg01.yml')
fslpython -c "import blm_serial; blm_serial.main('./BLM/test/cfg/test_cfg02.yml')
fslpython -c "import blm_serial; blm_serial.main('./BLM/test/cfg/test_cfg03.yml')
```
