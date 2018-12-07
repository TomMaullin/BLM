import sys
import scipy.ndimage
import nibabel as nib
import numpy as np
import subprocess
import warnings
import resource
import blm_setup
import blm_batch
import blm_concat

def main(Y_files, X): #flag='spat'

    print('Setting up analysis...')
    # Run the setup job to obtain the number of batches needed.
    blm_setup.main()

    # Load in inputs
    with open('blm_defaults.yml', 'r') as stream:
        inputs = yaml.load(stream)

    # Retrieve Output directory
    OutDir = inputs['outdir']

    # Load nB
    with open(os.path.join(OutDir, "nb.txt"), 'w') as f:
        nB = int(f.readline())

    # Run batch jobs
    for i in range(0, nB):
        print('Running job ' + str(i) + '/' + str(nB))
        blm_batch.main(i+1)

    # Run concatenation job
    print('Combining batch results...')
    blm_concat.main()

    print('Distributed analysis complete. Please see "' + outDir + "for output.")

if __name__ == "__main__":
    main()
