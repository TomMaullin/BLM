import os
import numpy
import pandas
import dask
import subprocess
from scipy import ndimage
from dask.distributed import Client, as_completed
from dask.distributed import performance_report
import nibabel as nib
import sys  

# Get the directory containing the script
script_dir = os.path.dirname(__file__)

# Add BLM and test directory to file path
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(script_dir))
sys.path.insert(0, '/well/nichols/users/inf852/BLM_lm_tests/test/')
sys.path.insert(0, '/well/nichols/users/inf852/BLM_lm_tests/')

# Import test generation and cleanup
from generate_test_data import *
from cleanup import cleanup
from blm_cluster import _main as blm
# from r_job import run_voxel_batch_in_R

def _main(argv=None):
    """
    Main function for running the BLM-lm test pipeline.

    Parameters:
    argv (list): List of command line arguments (not used).

    Returns:
    None
    """
    
    # -----------------------------------------------------------------
    # Temporary inputs
    # -----------------------------------------------------------------

    # Example usage
    sim_ind = 0
    num_voxel_batches = 200
    test_dir = "/well/nichols/users/inf852/BLM_lm_tests/test/"
    out_dir = "/well/nichols/users/inf852/BLM_lm_tests/output/"
    n = 100
    dim = np.array([100,100,100])

    # -----------------------------------------------------------------
    # Dask setup
    # -----------------------------------------------------------------

    # Number of nodes
    numNodes = 100

    # Cluster Type
    clusterType = 'SLURM'

    # -----------------------------------------------------------------
    # Create folders for simulation
    # -----------------------------------------------------------------

    # Make output directory
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Make simulation directory
    sim_dir = os.path.join(out_dir, 'sim' + str(sim_ind))
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)

    # Make new data directory.
    if not os.path.exists(os.path.join(sim_dir,"data")):
        os.mkdir(os.path.join(sim_dir,"data"))
        
    # Make new log directory.
    if not os.path.exists(os.path.join(sim_dir,"simlog")):
        os.mkdir(os.path.join(sim_dir,"simlog"))

    # -----------------------------------------------------------------
    # Generate data
    # -----------------------------------------------------------------

    generate_data(n, dim, out_dir, sim_ind)

    # -----------------------------------------------------------------
    # Run BLM
    # -----------------------------------------------------------------

    # Import inputs file
    inputs_yml = os.path.join(out_dir, 'sim'+ str(sim_ind), 'inputs.yml')

    # Run blm_cluster
    blm([inputs_yml])

    # -----------------------------------------------------------------
    # Set up cluster
    # -----------------------------------------------------------------

    # Check if we are using a HTCondor cluster
    if clusterType.lower() == 'htcondor':

        # Load the HTCondor Cluster
        from dask_jobqueue import HTCondorCluster
        cluster = HTCondorCluster()

    # Check if we are using an LSF cluster
    elif clusterType.lower() == 'lsf':

        # Load the LSF Cluster
        from dask_jobqueue import LSFCluster
        cluster = LSFCluster()

    # Check if we are using a Moab cluster
    elif clusterType.lower() == 'moab':

        # Load the Moab Cluster
        from dask_jobqueue import MoabCluster
        cluster = MoabCluster()

    # Check if we are using a OAR cluster
    elif clusterType.lower() == 'oar':

        # Load the OAR Cluster
        from dask_jobqueue import OARCluster
        cluster = OARCluster()

    # Check if we are using a PBS cluster
    elif clusterType.lower() == 'pbs':

        # Load the PBS Cluster
        from dask_jobqueue import PBSCluster
        cluster = PBSCluster()

    # Check if we are using an SGE cluster
    elif clusterType.lower() == 'sge':

        # Load the SGE Cluster
        from dask_jobqueue import SGECluster
        cluster = SGECluster()

    # Check if we are using a SLURM cluster
    elif clusterType.lower() == 'slurm':

        # Load the SLURM Cluster
        from dask_jobqueue import SLURMCluster
        cluster = SLURMCluster()

    # Check if we are using a local cluster
    elif clusterType.lower() == 'local':

        # Load the Local Cluster
        from dask.distributed import LocalCluster
        cluster = LocalCluster()

    # Raise a value error if none of the above
    else:
        raise ValueError('The cluster type, ' + clusterType + ', is not recognized.')

    # --------------------------------------------------------------------------------
    # Connect to client
    # --------------------------------------------------------------------------------

    # Connect to cluster
    client = Client(cluster)   

    # --------------------------------------------------------------------------------
    # Run R Jobs
    # --------------------------------------------------------------------------------

    # Ask for numNodes nodes for BLM batch
    cluster.scale(numNodes)

    # Empty futures list
    futures = []

    # Submit jobs
    for i in np.arange(num_voxel_batches):
        
        # Run the jobNum^{th} job.
        future_r = client.submit(run_voxel_batch_in_R, sim_ind, dim, i, num_voxel_batches,
                                 out_dir, test_dir, pure=False)

        # Append to list 
        futures.append(future_r)
    
    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    # Delete the future objects (NOTE: this is important - if you
    # don't delete these they get rerun every time).
    del i, completed, futures, future_r

    # --------------------------------------------------------------------------------
    # Run cleanup and print results
    # --------------------------------------------------------------------------------

    # Cleanup and results function
    cleanup(out_dir,sim_ind)


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

    import subprocess
    import sys  
    sys.path.insert(0, '/well/nichols/users/inf852/BLM_lm_tests/test/')
    sys.path.insert(0, '/well/nichols/users/inf852/BLM_lm_tests/')

    from generate_test_data import Rpreproc
    from cleanup import Rcleanup


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



if __name__ == "__main__":
    _main(sys.argv[1:])