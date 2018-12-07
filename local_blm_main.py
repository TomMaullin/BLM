import numpy as np
import yaml
import os
from BLM import blm_setup
from BLM import blm_batch
from BLM import blm_concat

def main():

    print('Setting up analysis...')
    # Run the setup job to obtain the number of batches needed.
    blm_setup.main()

    # Load in inputs
    with open(os.path.join(os.getcwd(),'blm_defaults.yml'), 'r') as stream:
        inputs = yaml.load(stream)

    # Retrieve Output directory
    OutDir = inputs['outdir']

    # Load nB
    with open(os.path.join(OutDir, "nb.txt"), 'r') as f:
        nB = int(f.readline())

    # Run batch jobs
    for i in range(0, nB):
        print('Running job ' + str(i+1) + '/' + str(nB))
        blm_batch.main(i+1)

    # Run concatenation job
    print('Combining batch results...')
    blm_concat.main()

    print('Distributed analysis complete. Please see "' + OutDir + "for output.")

if __name__ == "__main__":
    main()
