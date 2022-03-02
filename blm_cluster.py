from dask import config
from dask_jobqueue import SGECluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report
from lib.blm_setup import main1 as blm_setup
from lib.blm_batch import main2 as blm_batch
from lib.blm_concat import main3 as blm_concat
from lib.blm_concat2 import main3 as blm_concat2
from lib.blm_cleanup import main4 as blm_cleanup
import numpy as np

# Given a dask distributed client run BLM.
def main(cluster, client):

    # Inputs yaml
    inputs_yml = #...

    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Inputs file is first argument
    with open(os.path.join(ipath), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # --------------------------------------------------------------------------------
    # Read Output directory, work out number of batches
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']

    # --------------------------------------------------------------------------------
    # Clean up files
    # --------------------------------------------------------------------------------
    os.remove(os.path.join(OutDir, 'nb.txt'))
    if os.path.isdir(os.path.join(OutDir, 'tmp')):
        shutil.rmtree(os.path.join(OutDir, 'tmp'))


    # Need to return number of batches
    retnb = True

    # Ask for a node for setup
    cluster.scale(1)

    # Get number of batches
    future_0 = client.submit(blm_setup, inputs_yml, retnb, pure=False)
    nb = future_0.result()

    del future_0

    # Print number of batches
    print(nb)

    # Ask for 100 nodes for BLM batch
    cluster.scale(100)

    # Futures list
    futures = client.map(blm_batch, *[np.arange(nb)+1, [inputs_yml]*nb], pure=False)

    # # Wait for results
    # for future_b in as_completed(futures):
    #     future_b.result()

    # results
    results = client.gather(futures)
    del futures, results

    print('Batches completed')

    # --------------------------------------------------------
    # Plan
    # --------------------------------------------------------

    # --------------------------------------------------------
    # CONCAT
    # --------------------------------------------------------

    # Empty futures list
    futures = []

    # Loop through nodes
    for node in np.arange(1,101):

        # Give the i^{th} node the i^{th} partition of the data
        future_b = client.submit(blm_concat2, nb, node, 100, inputs_yml, pure=False)

        # Append to list
        futures.append(future_b)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    # Work out the max number of voxels we can actually compute at a time.
    # (This is really just a rule of thumb guess but works reasonably in
    # practice).
    nvb = MAXMEM/(10*8*(p**2))

    # We're now going to chunk up the analysis mask.
    nvb = np.min(2000, nvb)

    # Work out number of groups we have to split indices into.
    nvg = int(len(bamInds)//nvb+1)

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

   
    print('BLM code complete!')

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

    # Connect to cluster
    client = Client(cluster)   

    print('here4')

    # Run BLM
    main(cluster, client)

    print('here5')

    # Close the client
    client.close()