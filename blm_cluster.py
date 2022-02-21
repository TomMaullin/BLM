from dask import config
from dask_jobqueue import SGECluster
from distributed import Client, as_completed
from lib.blm_setup import main as blm_setup
from lib.blm_batch import main as blm_batch
from lib.blm_concat import main as blm_concat
from lib.blm_cleanup import main as blm_cleanup
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

    # Empty futures list
    futures = []

    # Run batch jobs
    for b in (np.arange(nb)+1):

        # Individual batch job
        future_b = client.submit(blm_batch, b, inputs_yml, pure=False)
        
        # Append to list
        futures.append(future_b)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for future_b in completed:
        future_b.result()

    print('Batches completed')

    # Ask for 1 node for BLM concat
    cluster.scale(1)

    # Concatenation job
    future_concat = client.submit(blm_concat, inputs_yml, pure=False)

    # Run concatenation job
    future_concat.result()

    print('Concat completed')

    # Run cleanup job
    future_cleanup = client.submit(blm_cleanup, inputs_yml, pure=False)

    # Run cleanup job
    future_cleanup.result()
    
    print('BLM code complete!')


# If running this function
if __name__ == "__main__":

    # timeouts
    config.set(distributed__comm__timeouts__tcp='90s')
    config.set(distributed__comm__timeouts__connect='90s')

    print('here1')

    with performance_report(filename="dask-report.html"):
        # Specify cluster setup
        cluster = SGECluster(cores=1,
                             memory="100GB",
                             queue='short.qc',
                             walltime='00:30:00',
                             extra=['--no-dashboard'],
                             interface="ib0",
                             local_directory="/well/nichols/users/inf852/BLMdask/")

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

        print(cluster.job_script())
