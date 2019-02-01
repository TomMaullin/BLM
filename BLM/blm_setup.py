import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import shutil
import yaml
import time
from BLM/blm_eval import blm_eval

def main(*args):

    t1 = time.time()

    # Change to blm directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if len(args)==0:
        # Load in inputs
        with open(os.path.join('..','blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        # In this case inputs is first argument
        inputs = args[0]      

    MAXMEM = eval(inputs['MAXMEM'])
    OutDir = inputs['outdir']

    # Make output directory and tmp
    if not os.path.isdir(OutDir):
        os.mkdir(OutDir)
    if not os.path.isdir(os.path.join(OutDir, "tmp")):
        os.mkdir(os.path.join(OutDir, "tmp"))

    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load in one nifti to check NIFTI size
    try:
        Y0 = nib.load(Y_files[0])
    except Exception as error:
        raise ValueError('The NIFTI "' + Y_files[0] + '"does not exist')

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTIsize = sys.getsizeof(np.zeros(Y0.shape,dtype='uint64'))

    if NIFTIsize > MAXMEM:
        raise ValueError('The NIFTI "' + Y_files[0] + '"is too large')

    # Load affine
    d0 = Y0.get_data()
    Y0aff = Y0.affine

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTIsize = sys.getsizeof(np.zeros(d0.shape,dtype='uint64'))

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use.
    blksize = np.floor(MAXMEM/8/NIFTIsize);
    if blksize == 0:
        raise ValueError('Blocksize too small.')

    # Check F contrast ranks 
    n_c = len(inputs['contrasts'])
    for i in range(0,n_c):

        if inputs['contrasts'][i]['c' + str(i+1)]['statType'] == 'F':

            # Read in contrast vector
            # Get number of parameters
            cvec = blm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
            cvec = np.array(cvec)
                
            # Get dimension of cvector
            q = cvec.shape[0]

            if np.linalg.matrix_rank(cvec)<q:
                raise ValueError('F contrast: \n' + str(cvec) + '\n is not of correct rank.')


    if len(args)==0:
        with open(os.path.join(OutDir, "nb.txt"), 'w') as f:
            print(int(np.ceil(len(Y_files)/int(blksize))), file=f)
    else:
        return(int(np.ceil(len(Y_files)/int(blksize))))

    w.resetwarnings()

if __name__ == "__main__":
    main()
