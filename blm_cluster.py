from dask import config
from dask_jobqueue import SGECluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report
from lib.blm_setup import main1 as blm_setup
from lib.blm_batch import main2 as blm_batch
from lib.blm_concat import main3 as blm_concat
from lib.blm_cleanup import main4 as blm_cleanup
import numpy as np

# Given a dask distributed client run BLM.
def main(cluster, client):

    # Inputs yaml
    inputs_yml = #...

    # Need to return number of batches
    retnb = True

    # Ask for a node for setup
    cluster.scale(1)

    # Get number of batches
    future_0 = client.submit(blm_setup, inputs_yml, retnb, pure=False)
    nb = future_0.result()

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
    del results

    print('Batches completed')

    # # Ask for 1 node for BLM concat
    # cluster.scale(1)

    # # Concatenation job
    # future_concat = client.submit(blm_concat, inputs_yml, pure=False)

    # # Run concatenation job
    # future_concat.result()

    # print('Concat completed')

    # # Run cleanup job
    # future_cleanup = client.submit(blm_cleanup, inputs_yml, pure=False)

    # # Run cleanup job
    # future_cleanup.result()
    
    # print('BLM code complete!')


# If running this function
if __name__ == "__main__":

    # timeouts
    config.set(distributed__comm__timeouts__tcp='90s')
    config.set(distributed__comm__timeouts__connect='90s')

    print('here1')

    # Specify cluster setup
    cluster = SGECluster(cores=1,
                         memory="100GB",
                         queue='short.qc',
                         walltime='00:30:00',
                         interface="ib0",
                         local_directory="/well/nichols/users/inf852/BLMdask/",
                         scheduler_options={'dashboard_address': ':8888'})

    print('here2')

    print('here3')

    # Connect to cluster
    client = Client(cluster)   

    with performance_report(filename="well/nichols/users/inf852/BLMdask/dask-report.html"):
        print('here4')

        # Run BLM
        main(cluster, client)

        print('here5')

        # Close the client
        client.close()

        print(cluster.job_script())
