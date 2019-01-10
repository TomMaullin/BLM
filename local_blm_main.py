import numpy as np
import yaml
import os
from BLM import blm_setup
from BLM import blm_batch
from BLM import blm_concat

def main(*args):

    print('Setting up analysis...')

    if len(args)==0:
        # Load in inputs
        with open(os.path.join(os.getcwd(),'blm_defaults.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        # In this case inputs is first argument
        inputs = args[0]

    # Run the setup job to obtain the number of batches needed.
    nB = blm_setup.main(inputs)

    # Retrieve Output directory
    OutDir = inputs['outdir']

    # Load nB
    with open(os.path.join(OutDir, "nb.txt"), 'r') as f:
        nB = int(f.readline())

    # Run batch jobs and concatenate results
    print('Running batch 1/' + str(nB))
    sumXtX, sumXtY, sumYtY, sumnmap = blm_batch.main(1, inputs)
    for i in range(1, nB):
        print('Running batch ' + str(i+1) + '/' + str(nB))
        XtX, XtY, YtY, nmap = blm_batch.main(i+1, inputs)
        sumXtX = sumXtX + XtX
        sumXtY = sumXtY + XtY
        sumYtY = sumYtY + YtY
        sumnmap = sumnmap + nmap

    # Run concatenation job
    print('Combining batch results...')
    blm_concat.main(inputs, sumXtX, sumXtY, sumXtY, sumnmap)

    print('Distributed analysis complete. Please see "' + OutDir + '" for output.')

if __name__ == "__main__":
    main()
