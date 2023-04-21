import subprocess
import sys  
sys.path.insert(0, '/well/nichols/users/inf852/BLM_lm_tests/test/')
sys.path.insert(0, '/well/nichols/users/inf852/BLM_lm_tests/')

from generate_test_data import Rpreproc
from cleanup import Rcleanup


def run_voxel_batch_in_R(sim_ind, dim, batch_no, num_voxel_batches, out_dir, test_dir):
    """
    Preprocesses a batch of voxel data and runs parameter estimation in R.

    Parameters:
    sim_ind (int): Index of the simulation.
    dim (int): Number of dimensions in the voxel data.
    batch_no (int): Index of the current batch.
    num_voxel_batches (int): Total number of batches to split the voxel data into.
    out_dir (str): Directory to save the output files.
    test_dir (str): Directory containing the R scripts for parameter estimation.

    Returns:
    None
    """

    print('here4')
    
    
    print('here6')
    # Preprocess this batch
    Rpreproc(out_dir, sim_ind, dim, num_voxel_batches, batch_no)
    
    print('here7')
    # Load R and run parameter estimation in a single command
    r_path = "/apps/well/R/3.4.3/bin/R"

    print('here8')
    # Write the R command to run the job
    r_cmd = (
        f"module load R/3.4.3 && "
        f"{r_path} CMD BATCH --no-save --no-restore "
        f"'--args simInd={sim_ind} batchNo={batch_no} outDir=\"'{out_dir}'\"' "
        f"{test_dir}/lm_paramEst.R "
        f"{out_dir}/sim{sim_ind}/simlog/Rout{sim_ind}_{batch_no}.txt"
    )

    print('here9')
    # Run the job
    subprocess.run(r_cmd, shell=True)
    
    print('here10')
    # Cleanup job in R to combine files
    Rcleanup(out_dir, sim_ind, num_voxel_batches, batch_no)
