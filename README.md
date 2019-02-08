# BLM-py
This repository contains all code for the python implementation of distributed OLS for locally stored data.

## Requirements
To use the BLM-py code, `fsl 5.0.10` or greater must be installed and `fslpython` must be configured correctly. Alternatively the following python packages must be installed:

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

## Testing

To generate test cases:

```
bash ./generate_test_cases.sh
```

To check the logs:

```
bash ./check_logs.sh
```

To verify the test cases against ground truth:

```
bash ./verify_test_cases.sh $GTDIR
```

(Where `$GTDIR` is a directory containing ground truth data from a previous run, i.e. inside `$GTDIR` are the folders `test_cfg1`, `test_cfg2`, ... ect).
