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
	nb = client.submit(blm_setup, inputs_yml, retnb)
	nb = nb.result()

# If running this function
if __name__ == "__main__":

	# Specify cluster setup
	cluster = SGECluster(cores=24,
	                     memory="100GB",
	                     queue='short.qc',
	                     walltime='02:00:00')


	# Start 100 workers in 100 jobs that match the description above
	cluster.scale(100)  

	# Connect to cluster
	client = Client(cluster)   

	# Run BLM
	main(client)

	# Close the client
	client.close()

	# Print errors
	print(cluster.job_script())