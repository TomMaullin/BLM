import numpy as np
import yaml
import os
from BLM import blm_setup
from BLM import blm_batch
from BLM import blm_concat

def main(**args):

    print('Setting up analysis...')

    if len(args)==0:
        # Load in inputs
        with open(os.path.join(os.getcwd(),'blm_defaults.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        # In this case inputs is first argument
        inputs = args[0]

    # Run the setup job to obtain the number of batches needed.
    blm_setup.main(inputs)

    # Retrieve Output directory
    OutDir = inputs['outdir']

    # Load nB
    with open(os.path.join(OutDir, "nb.txt"), 'r') as f:
        nB = int(f.readline())

    # Run batch jobs
    for i in range(0, nB):
        print('Running batch ' + str(i+1) + '/' + str(nB))
        blm_batch.main(i+1, inputs)

    # Run concatenation job
    print('Combining batch results...')
    blm_concat.main(inputs)

    print('Distributed analysis complete. Please see "' + OutDir + '" for output.')

if __name__ == "__main__":
    main()
