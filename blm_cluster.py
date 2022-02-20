from dask_jobqueue import SGECluster
from dask.distributed import Client
from lib.blm_setup import main as blm_setup
from lib.blm_batch import main as blm_batch
from lib.blm_concat import main as blm_concat
from lib.blm_cleanup import main as blm_cleanup

# Given a dask distributed client run BLM.
def main(client):

	# Inputs yaml
	inputs_yml = #...

	# Need to return number of batches
	retnb = True

	# Get number of batches
	nb = client.submit(blm_setup, inputs_yml, retnb, pure=False)
	nb = nb.result()

	# Print number of batches
	print(nb)

# If running this function
if __name__ == "__main__":

	print('here1')

	# Specify cluster setup
	cluster = SGECluster(cores=1,
	                     memory="100GB",
	                     queue='short.qc',
	                     walltime='01:00:00',
	                     extra=['--no-dashboard'])

	print('here2')

	# Set maximum number of jobs
	cluster.adapt(maximum_jobs=20)

	print('here3')

	# Connect to cluster
	client = Client(cluster)   

	print('here4')

	# Run BLM
	main(client)

	print('here5')

	# Close the client
	client.close()

	# Print errors
	print(cluster.job_script())