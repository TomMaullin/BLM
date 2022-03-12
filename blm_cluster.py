from dask import config
from dask_jobqueue import SGECluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report
from lib.blm_setup import main1 as blm_setup
from lib.blm_batch import main2 as blm_batch
from lib.blm_concat2 import main3 as blm_concat2
from lib.blm_concat2 import combineUniqueAtB as blm_concat3
from lib.blm_results2 import main3 as blm_results2
from lib.blm_cleanup import main4 as blm_cleanup
from lib.fileio import pracNumVoxelBlocks
import numpy as np
import os
import shutil
import yaml

# Given a dask distributed client run BLM.
def main(cluster):

    # Inputs yaml
    inputs_yml = #...

    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Inputs file is first argument
    with open(inputs_yml, 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # --------------------------------------------------------------------------------
    # Read Output directory, work out number of batches
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']

    # Need to return number of batches
    retnb = True

    # Connect to cluster
    client = Client(cluster)   

    # Ask for a node for setup
    cluster.scale(1)

    # Get number of batches
    future_0 = client.submit(blm_setup, inputs_yml, retnb, pure=False)
    nb = future_0.result()

    del future_0

    # MARKER: ASK USER ABOUT PREVIOUS OUTPUT

    # Print number of batches
    print(nb)

    # Ask for 100 nodes for BLM batch
    cluster.scale(100)

    # Futures list
    futures = client.map(blm_batch, *[np.arange(nb)+1, [inputs_yml]*nb], pure=False)

    # results
    results = client.gather(futures)
    del futures, results

    print('Batches completed')

    # --------------------------------------------------------
    # CONCAT
    # --------------------------------------------------------

    # Batch jobs
    maskJob = False

    # Groups of files
    fileGroups = np.array_split(np.arange(nb)+1, 100)

    # Empty futures list
    futures = []

    # Loop through nodes
    for node in np.arange(1,100 + 1):

        # Run the jobNum^{th} job.
        future_c = client.submit(blm_concat3, 'XtX', OutDir, fileGroups[node-1], pure=False)

        # Append to list
        futures.append(future_c)

    # Loop through nodes
    for node in np.arange(1,100 + 1):

        # Give the i^{th} node the i^{th} partition of the data
        future_b = client.submit(blm_concat2, nb, node, 100, maskJob, inputs_yml, pure=False)

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

    print('hereeee')
    print(nb, 101, 100, maskJob, inputs_yml)

    # The first job does the analysis mask (this is why the 3rd argument is set to true)
    future_b_first = client.submit(blm_concat2, nb, 101, 100, maskJob, inputs_yml, pure=False)
    res = future_b_first.result()

    del future_b_first, res


    # # --------------------------------------------------------
    # # AtB
    # # --------------------------------------------------------
    # # Empty futures list
    # futures = []

    # # Loop through nodes
    # for jobNum in np.arange(np.minimum(100,nb)): # MARKER

    # # Completed jobs
    # completed = as_completed(futures)

    # # Wait for results
    # for i in completed:
    #     i.result()

    # del i, completed, futures, future_c

    # print('AtB run')

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    # Number of jobs for results (practical number of voxel batches)
    pnvb = int(np.maximum(100*10, pracNumVoxelBlocks(inputs)))

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

    #
    # --------------------------------------------------------
    # # Ask for 1 node for BLM concat
    # cluster.scale(1)

    # # Concatenation job
    # future_concat = client.submit(blm_concat, nb, inputs_yml, pure=False)

    # print('0')

    # # Run concatenation job
    # future_concat.result()
    # del future_concat

    # print('1')

    # client.recreate_error_locally(future_concat) 

    # print(client.recreate_error_locally(future_concat)) 

    # print('2')

    # print('Concat completed')


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

    # timeouts
    config.set(distributed__comm__timeouts__tcp='90s')
    config.set(distributed__comm__timeouts__connect='90s')
    config.set(scheduler='single-threaded')
    config.set({'distributed.scheduler.allowed-failures': 50}) 
    # config.set({'distributed.scheduler.work-stealing': False}) 
    config.set(admin__tick__limit='3h')


    print('here1')

    # Specify cluster setup
    cluster = SGECluster(cores=1,
                         memory="100GB",
                         queue='short.qc',
                         walltime='01:00:00',
                         interface="ib0",
                         local_directory="/well/nichols/users/inf852/BLMdask/",
                         log_directory="/well/nichols/users/inf852/BLMdask/log/",
                         silence_logs=False,
                         scheduler_options={'dashboard_address': ':8889'})


    print('here2')

    print('here3')

    print('here4')

    # Run BLM
    main(cluster)

    print('here5')
