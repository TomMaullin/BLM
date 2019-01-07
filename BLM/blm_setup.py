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

def main(*args):

    t1 = time.time()

    # Change to blm directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if len(args)==0:
        # Load in inputs
        with open(os.path.join('..','blm_defaults.yml'), 'r') as stream:
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

    d0 = Y0.get_data()
    Y0aff = Y0.affine

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTIsize = sys.getsizeof(np.zeros(d0.shape,dtype='uint64'))

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use.
    blksize = np.floor(MAXMEM/8/NIFTIsize);
    
    with open(os.path.join(OutDir, "nb.txt"), 'w') as f:
        print(int(np.ceil(len(Y_files)/int(blksize))), file=f)

    w.resetwarnings()

    t2 = time.time()

if __name__ == "__main__":
    main()
