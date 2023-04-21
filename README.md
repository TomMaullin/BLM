# BLM-py
This repository contains the code for Big Linear Models for Neuroimaging cluster and local usage.

## Requirements
To use the BLM-py code, please clone this repository to your cluster. 

```
git clone https://github.com/TomMaullin/BLM.git
```

Then pip install the requirements:

```
pip install -r requirements.txt
```

Finally, you must set up your `dask-jobqueue` configuration file, which is likely located at `~/.config/dask/jobqueue.yaml`. This will require you to provide some details about your HPC system. See [here](https://jobqueue.dask.org/en/latest/configuration-setup.html#managing-configuration-files) for further detail. For instance, if you are using rescomp your `jobqueue.yaml` file may look something like this:

```
jobqueue:
  slurm:
    name: dask-worker

    # Dask worker options
    cores: 1                 # Total number of cores per job
    memory: "100GB"                # Total amount of memory per job
    processes: 1                # Number of Python processes per job

    interface: ib0             # Network interface to use like eth0 or ib0
    death-timeout: 60           # Number of seconds to wait if a worker can not find a scheduler
    local-directory: "/path/of/your/choosing/"       # Location of fast local storage like /scratch or $TMPDIR
    log-directory: "/path/of/your/choosing/"
    silence_logs: True

    # SLURM resource manager options
    shebang: "#!/usr/bin/bash"
    queue: short
    project: null
    walltime: '01:00:00'
    job-cpu: null
    job-mem: null
    log-directory: null

    # Scheduler options
    scheduler-options: {'dashboard_address': ':46405'}
```


## Usage
To run `BLM-py` first specify your design using `blm_config.yml` and then run the job in parallel by following the below guidelines.

### Specifying your model
The regression model for BLM must be specified in `blm_config.yml`. Below is a complete list of possible inputs to this file.

#### Mandatory fields
The following fields are mandatory:

 - `Y_files`: Text file containing a list of response variable images in NIFTI format.
 - `X`: CSV file of the design matrix (no column header, no ID row).
 - `outdir`: Path to the output directory.
 - `contrasts`: Contrast vectors to be tested. They should be listed as `c1,c2,...` etc and each contrast should contain the fields:
   - `name`: A name for the contrast. i.e. `AwesomelyNamedContrast1`.
   - `vector`: A vector for the contrast. This contrast must be one dimensional for a T test and two dimensional for an F test. For example; `[1, 0, 0]` (T contrast) or `[[1, 0, 0],[0,1,0]]` (F contrast).
 - `clusterType`: Cluster type the user wants to use (e.g. SLURM, SGE, etc).
 - `numNodes`: Number of nodes to use for computation.
 
   At least one contrast must be given, see `Examples` for an example of how to specify contrasts.
 
#### Optional fields

The following fields are optional:

 - `MAXMEM`: This is the maximum amount of memory (in bits) that the BLM code is allowed to work with. How this should be set depends on your machine capabilities; the default value however matches the SPM default of 2^32 (note this must be in python notation i.e. `2**32`).
 - `data_mask_files`: A text file containing a list of masks to be applied to the `Y_files`. 
   - The number of masks must be equal to the number of `Y_files` as each mask is applied to the corresponding entry `Y_files`. E.g. The first mask listed for `data_mask_files` will be applied to the first nifti in `Y_files`, the second mask in `data_mask_files` will be applied to the second nifti in `Y_files` and so on. 
 - `Missingness`: This field allows the user to mask the image based on how many studies had recorded values for each voxel. This can be specified in 3 ways.
   - `MinPercent`: The percentage of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `0.1` then any voxel with recorded values for at least 10% of studies will be kept in the analysis.
   - `MinN`: The number of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `20` then any voxel with recorded values for at least 20 studies will be kept in the analysis.
 - `analysis_mask`: A mask to be applied during analysis.
 - `OutputCovB`: If set to `True` this will output between beta covariance maps. For studies with a large number of paramters this may not be desirable as, for example, 30 analysis paramters will create 30x30=900 between beta covariance maps. By default this is set to `True`.
 - `data_mask_thresh`: Any voxel with value below this threshold will be treated as missing data. (By default, no such thresholding  is done, i.e. `data_mask_thresh` is essentially -infinity). 
 - `minlog`: Any `-inf` values in the `-log10(p)` maps will be converted to the value of `minlog`. Currently, a default value of `-323.3062153431158` is used as this is the most negative value which was seen during testing before `-inf` was encountered (see [this thread](https://github.com/TomMaullin/BLM/issues/76) for more details).


 
#### Examples

Below are some example `blm_config.yml` files.

Example 1: A minimal configuration.

```
Y_files: /path/to/data/Y.txt
X: /path/to/data/X.csv
outdir: /path/to/output/directory/
contrasts:
  - c1:
      name: Tcontrast1
      vector: [1, 0, 1, 0, 1]
clusterType: SLURM
numNodes: 100
```

Example 2: A configuration with multiple optional fields.

```
MAXMEM: 2**32
Y_files: /path/to/data/Y.txt
data_mask_files: /path/to/data/M_.txt
data_mask_thresh: 0.1
X: /path/to/data/X.csv
outdir: /path/to/output/directory/
contrasts:
  - c1:
      name: Tcontrast1
      vector: [1, 0, 0]
  - c2:
      name: Tcontrast2
      vector: [0, 1, 0]
  - c3:
      name: Tcontrast3
      vector: [0, 0, 1]
  - c4:
      name: Fcontrast1
      vector: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
Missingness:
  MinPercent: 0.10
  MinN: 15
analysis_mask: /path/to/data/MNI152_T1_2mm_brain_mask.nii.gz
clusterType: SLURM
numNodes: 100
```

### Running the Analysis

On your HPC system, ensure you are in the `BLM-py` directory and once you are happy with the analysis you have specified in `blm_config.yml`, run the following command:

```
python blm_cluster.py &
```

You can watch your analysis progress either by using `qstat` or `squeue` (depending on your system), or by using the interactive dask console. To do so, in a seperate terminal, tunnel into your HPC as follows:

```
ssh -L <local port>:localhost:<remote port> username@hpc_address
```

where the local port is the port you want to view on your local machine and the remote port is the dask dashboard adress (for instance, if you are on rescomp and you used the above `jobqueue.yaml`, `<remote port>` is `46405`). On your local machine, in a browser you can now go to `http://localhost:<local port>/` to watch the analysis run.

### Analysis Output

Below is a full list of NIFTI files output after a BLM analysis.

| Filename  | Description  |
|---|---|
| `blm_vox_mask` | This is the analysis mask. |
| `blm_vox_n` | This is a map of the number of subjects which contributed to each voxel in the final analysis. |
| `blm_vox_edf` | This is the spatially varying error degrees of freedom mask. |
| `blm_vox_beta`  | These are the beta estimates.  |
| `blm_vox_con`  | These are the contrasts multiplied by the estimate of beta (this is the same as `COPE` in FSL).  |
| `blm_vox_cov`  | These are the between-beta covariance estimates.  |
| `blm_vox_conSE` | These are the standard error of the contrasts multiplied by beta (only available for T contrasts). |
| `blm_vox_conR2` | These are the partial R^2 maps for the contrasts (only available for F contrasts). |
| `blm_vox_resms` | This is the residual mean squares map for the analysis. |
| `blm_vox_conT` | These are the T statistics for the contrasts (only available for T contrasts). |
| `blm_vox_conF` | These are the F statistics for the contrasts (only available for F contrasts). |
| `blm_vox_conTlp` | These are the maps of -log10 of the uncorrected P values for the contrasts (T contrast). |
| `blm_vox_conFlp` | These are the maps of -log10 of the uncorrected P values for the contrasts (F contrast). |

The maps are given the same ordering as the inputs. For example, in `blm_vox_con`, the `0`th volume corresponds to the `1`st contrast, the `1`st volume corresponds to the `2`nd contrast and so on. For covariances, the ordering is of a similar form with covariance between beta 1 and  beta 1 (variance of beta 1) being the `0`th volume, covariance between beta 1 and  beta 2 being the `1`st volume and so on. In addition, a copy of the design is saved in the output directory as `inputs.yml`. It is recommended that this be kept for data provenance purposes.

## Testing

The current BLM tests run on rescomp and compare BLM's performance to that of looping over voxels in R. Details on how to run the tests are given below.

### Usage

To run the tests, navigate to the main BLM directory and use the following command in the commandline:

```
python ./test/blm_cluster_test.py --sim_ind [SIM_IND] --num_nodes [NUM_NODES] --out_dir [OUT_DIR] [--clusterType [CLUSTER_TYPE]]
```


### Arguments

- `--sim_ind [SIM_IND]`: (Required) An integer value between 1 and 4. Determines the values of `n` (number of observations) and `num_voxel_batches` (number of batches to run the voxelwise R analysis in) used in the test.

  - `sim_ind = 1`: `n = 100` observations, `num_voxel_batches = 100` batches
  - `sim_ind = 2`: `n = 200` observations, `num_voxel_batches = 200` batches
  - `sim_ind = 3`: `n = 500` observations, `num_voxel_batches = 200` batches
  - `sim_ind = 4`: `n = 1000` observations, `num_voxel_batches = 500` batches

- `--num_nodes [NUM_NODES]`: (Optional) An integer value representing the number of nodes for the Dask setup. Defaults to 100.

- `--out_dir [OUT_DIR]`: (Required) A string representing the output directory path.

- `--clusterType [CLUSTER_TYPE]`: (Optional) A string representing the cluster type for the Dask setup. Defaults to "slurm".

### Example

To run the script with a simulation index of 1, 100 nodes, an output directory of "/path/to/blm/BLM_lm_tests/output/", and the default cluster type, use the following command:


```
./test/blm_cluster_test.py --sim_ind 1 --num_nodes 100 --out_dir "/path/to/blm/BLM_lm_tests/output/"
```