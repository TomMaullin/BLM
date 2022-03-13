from dask import config
from dask_jobqueue import SGECluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report
from lib.blm_setup import setup
from lib.blm_batch import compute_product_forms
from lib.blm_concat import combine_batch_masking, combine_batch_designs
from lib.blm_results import output_results
from lib.fileio import pracNumVoxelBlocks
import numpy as np
import os
import sys
import shutil
import yaml

# Given a dask distributed client run BLM.
def main(cluster, inputs):

    # --------------------------------------------------------------------------------
    # Read Output directory, work out number of batches
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']

    # Get number of nodes
    numNodes = inputs['numNodes']

    # Need to return number of batches
    retnb = True

    # Connect to cluster
    client = Client(cluster)   

    # Ask for a node for setup
    cluster.scale(1)

    # Get number of batches
    future_0 = client.submit(setup, inputs_yml, retnb, pure=False)
    nb = future_0.result()

    del future_0

    # MARKER: ASK USER ABOUT PREVIOUS OUTPUT

    # Ask for numNodes nodes for BLM batch
    cluster.scale(numNodes)

    # Futures list
    futures = client.map(compute_product_forms, *[np.arange(nb)+1, [inputs_yml]*nb], pure=False)

    # results
    results = client.gather(futures)
    del futures, results

    print('Batches completed')

    # --------------------------------------------------------
    # CONCATENATE NUMBER OF OBSERVATIONS AND DESIGNS
    # --------------------------------------------------------

    # Batch jobs
    maskJob = False

    # Groups of files
    fileGroups = np.array_split(np.arange(nb)+1, numNodes)

    # Empty futures list
    futures = []

    # Loop through nodes
    for node in np.arange(1,numNodes + 1):

        # Run the jobNum^{th} job.
        future_c = client.submit(blm_concat3, 'XtX', OutDir, fileGroups[node-1], pure=False)

        # Append to list
        futures.append(future_c)

    # Loop through nodes
    for node in np.arange(1,numNodes + 1):

        # Give the i^{th} node the i^{th} partition of the data
        future_b = client.submit(blm_concat2, nb, node, numNodes, maskJob, inputs_yml, pure=False)

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
    future_b_first = client.submit(blm_concat2, nb, numNodes + 1, numNodes, maskJob, inputs_yml, pure=False)
    res = future_b_first.result()

    del future_b_first, res

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    # Number of jobs for results (practical number of voxel batches)
    pnvb = int(np.maximum(numNodes, pracNumVoxelBlocks(inputs)))

    # Empty futures list
    futures = []

    # Loop through nodes
    for jobNum in np.arange(pnvb):

        # Run the jobNum^{th} job.
        future_c = client.submit(blm_results2, jobNum, pnvb, nb, inputs_yml, pure=False)

        # Append to list
        futures.append(future_c)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    del i, completed, futures, future_c

    # --------------------------------------------------------------------------------
    # Clean up files
    # --------------------------------------------------------------------------------
    if os.path.isfile(os.path.join(OutDir, 'nb.txt')):
        os.remove(os.path.join(OutDir, 'nb.txt'))
    if os.path.isdir(os.path.join(OutDir, 'tmp')):
        shutil.rmtree(os.path.join(OutDir, 'tmp'))
   
    print('BLM analysis complete!')
    print('')
    print('---------------------------------------------------------------------------')
    print('')
    print('Check results in: ', OutDir)

# If running this function
if __name__ == "__main__":

    # Inputs file is first argument
    if len(sys.argv)>1:
        if not os.path.split(sys.argv[1])[0]:
            inputs_yml = os.path.join(os.getcwd(),sys.argv[1])
        else:
            inputs_yml = sys.argv[1]
    else:
        inputs_yml = os.path.join(os.path.realpath(__file__),'blm_config.yml')

    # Read in inputs
    with open(inputs_yml, 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # Timeouts
    config.set(distributed__comm__timeouts__tcp='90s')
    config.set(distributed__comm__timeouts__connect='90s')
    config.set(scheduler='single-threaded')
    config.set({'distributed.scheduler.allowed-failures': 50}) 
    config.set(admin__tick__limit='3h')

    # Specify cluster setup
    cluster = SGECluster(cores=1,
                         memory="100GB",
                         queue='short.qc',
                         walltime='01:00:00',
                         interface="ib0",
                         local_directory="/well/nichols/users/inf852/BLMdask/",
                         log_directory="/well/nichols/users/inf852/BLMdask/log/",
                         silence_logs=True,
                         scheduler_options={'dashboard_address': ':8889'})


    # Run BLM
    main(cluster, inputs)

