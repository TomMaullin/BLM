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

 - `contrasts`

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
