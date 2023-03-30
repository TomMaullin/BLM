import os
import sys
import shutil
import yaml
import numpy as np
from lib.blm_setup import setup
from lib.blm_batch import compute_product_forms
from lib.blm_concat import combine_batch_masking, combine_batch_designs
from lib.blm_results import output_results
from lib.fileio import pracNumVoxelBlocks
from dask import config
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report


def _main(argv=None):
    
    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Inputs file is first argument
    if len(argv)<1:
        raise ValueError('Please provide an inputs YAML file.')
        
    # Get the inputs filepath
    else:
        inputs_yml = argv[0]
        
    # Load the inputs yaml file
    with open(inputs_yml, 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)
    
    # --------------------------------------------------------------------------------
    # Read Output directory, work out number of batches
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']
    
    # Get number of nodes
    numNodes = inputs['numNodes']

    # Need to return number of batches
    retnb = True

    # --------------------------------------------------------------------------------
    # Set up cluster
    # --------------------------------------------------------------------------------
    if 'clusterType' in inputs:

        # Check if we are using a HTCondor cluster
        if inputs['clusterType'].lower() == 'htcondor':

            # Load the HTCondor Cluster
            from dask_jobqueue import HTCondorCluster
            cluster = HTCondorCluster()

        # Check if we are using an LSF cluster
        elif inputs['clusterType'].lower() == 'lsf':

            # Load the LSF Cluster
            from dask_jobqueue import LSFCluster
            cluster = LSFCluster()

        # Check if we are using a Moab cluster
        elif inputs['clusterType'].lower() == 'moab':

            # Load the Moab Cluster
            from dask_jobqueue import MoabCluster
            cluster = MoabCluster()

        # Check if we are using a OAR cluster
        elif inputs['clusterType'].lower() == 'oar':

            # Load the OAR Cluster
            from dask_jobqueue import OARCluster
            cluster = OARCluster()

        # Check if we are using a PBS cluster
        elif inputs['clusterType'].lower() == 'pbs':

            # Load the PBS Cluster
            from dask_jobqueue import PBSCluster
            cluster = PBSCluster()

        # Check if we are using an SGE cluster
        elif inputs['clusterType'].lower() == 'sge':

            # Load the SGE Cluster
            from dask_jobqueue import SGECluster
            cluster = SGECluster()

        # Check if we are using a SLURM cluster
        elif inputs['clusterType'].lower() == 'slurm':

            # Load the SLURM Cluster
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster()

        # Check if we are using a local cluster
        elif inputs['clusterType'].lower() == 'local':

            # Load the Local Cluster
            from dask.distributed import LocalCluster
            cluster = LocalCluster()

        # Raise a value error if none of the above
        else:
            raise ValueError('The cluster type, ' + inputs['clusterType'] + ', is not recognized.')

    else:
        # Raise a value error if the cluster type was not specified
        raise ValueError('Please specify "clusterType" in the inputs yaml.')
    
    # --------------------------------------------------------------------------------
    # Connect to client
    # --------------------------------------------------------------------------------
    
    # Connect to cluster
    client = Client(cluster)   

    # --------------------------------------------------------------------------------
    # Run Setup
    # --------------------------------------------------------------------------------

    # Ask for a node for setup
    cluster.scale(1)

    # Get number of batches
    future_s = client.submit(setup, inputs_yml, retnb, pure=False)
    nb = future_s.result()

    # Delete the future object (NOTE: This is important! If you don't delete this dask
    # tries to rerun it every time you call the result function again, e.g. after each
    # stage of the pipeline).
    del future_s
    
    # --------------------------------------------------------------------------------
    # Run Batch Jobs
    # --------------------------------------------------------------------------------

    # Ask for numNodes nodes for BLM batch
    cluster.scale(numNodes)

    # Empty futures list
    futures = []

    # Submit jobs
    for i in np.arange(1,nb+1):

        # Run the jobNum^{th} job.
        future_b = client.submit(compute_product_forms, i, inputs_yml, pure=False)

        # Append to list 
        futures.append(future_b)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    # Delete the future objects (NOTE: see above comment in setup section).
    del i, completed, futures, future_b

    # --------------------------------------------------------------------------------
    # Run Concatenation Job
    # --------------------------------------------------------------------------------

    # Batch jobs
    maskJob = False

    # Groups of files
    fileGroups = np.array_split(np.arange(nb)+1, numNodes)

    # Check for empty filegroups
    fileGroups = [i for i in fileGroups if i.size!=0]

    # Number of file groups
    numFileGroups = len(fileGroups)

    # Empty futures list
    futures = []

    # Loop through nodes
    for node in np.arange(1,numFileGroups + 1):

        # Run the jobNum^{th} job.
        future_c = client.submit(combine_batch_designs, 'XtX', OutDir, fileGroups[node-1], pure=False)

        # Append to list 
        futures.append(future_c)

    # Loop through nodes
    for node in np.arange(1,numNodes + 1):

        # Give the i^{th} node the i^{th} partition of the data
        future_b = client.submit(combine_batch_masking, nb, node, numNodes, maskJob, inputs_yml, pure=False)

        # Append to list
        futures.append(future_b)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    del i, completed, futures, future_b, future_c

    # Mask job
    maskJob = True

    # The first job does the analysis mask (this is why the 3rd argument is set to true)
    future_b_first = client.submit(combine_batch_masking, nb, numNodes + 1, numNodes, maskJob, inputs_yml, pure=False)
    res = future_b_first.result()

    del future_b_first, res

    # --------------------------------------------------------------------------------
    # Run Results Jobs
    # --------------------------------------------------------------------------------

    # Number of jobs for results (practical number of voxel batches)
    pnvb = int(np.minimum(numNodes, pracNumVoxelBlocks(inputs)))

    print(pnvb, nb)
    with open(os.path.join(OutDir,"tmp.txt"), "w") as f:
        f.write(pnvb,nb)

    # Empty futures list
    futures = []

    # Loop through nodes
    for jobNum in np.arange(pnvb):

        # Run the jobNum^{th} job.
        future_c = client.submit(output_results, jobNum, pnvb, nb, inputs_yml, pure=False)

        # Append to list
        futures.append(future_c)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    del i, completed, futures, future_c

    # MARKER UP TO HERE

    # --------------------------------------------------------------------------------
    # Clean up files
    # --------------------------------------------------------------------------------
    if os.path.isfile(os.path.join(OutDir, 'nb.txt')):
        os.remove(os.path.join(OutDir, 'nb.txt'))

    print('BLM analysis complete!')
    print('')
    print('---------------------------------------------------------------------------')
    print('')
    print('Check results in: ', OutDir)

    # Close the client
    client.close()

if __name__ == "__main__":
    _main(sys.argv[1:])