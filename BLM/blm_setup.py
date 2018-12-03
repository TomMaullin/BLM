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

def main(*args):

    # Change to blm directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # If X and Y weren't given we look in defaults for all arguments.
    if len(args)<2:

        with open('blm_defaults.yml', 'r') as stream:
            inputs = yaml.load(stream)

        MAXMEM = eval(inputs['MAXMEM'])

        with open(inputs['Y_files']) as a:

            Y_files = []
            i = 0
            for line in a.readlines():

                Y_files.append(line.replace('\n', ''))

        X = np.loadtxt(inputs['X'], delimiter=',')

        SVFlag = inputs['SVFlag']

    # else Y_files is the first input and X is the second.
    elif len(args)==2:

        Y_files = args[0]
        X = args[1]

        with open('blm_defaults.yml', 'r') as stream:
            inputs = yaml.load(stream)
        MAXMEM = eval(inputs['MAXMEM'])
        SVFlag = inputs['SVFlag']

    # And MAXMEM may be the third input
    else:

        Y_files = args[0]
        X = args[1]
        MAXMEM = args[2]        

        with open('blm_defaults.yml', 'r') as stream:
            inputs = yaml.load(stream)
        SVFlag = inputs['SVFlag']

    # Load in one nifti to check NIFTI size
    try:
        Y0 = nib.load(Y_files[0])
    except Exception as error:
        raise ValueError('The NIFTI "' + Y_files[0] + '"does not exist')

    d0 = Y0.get_data()
    Y0aff = Y0.affine

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTIsize = sys.getsizeof(np.zeros(d0.shape,dtype='uint64'))

    # Count number of scans contributing to voxels
    sumVox = np.zeros(d0.shape)
    print(repr(sumVox))
    print(repr(sum(sum(sumVox))))

    # Initial checks for NIFTI compatability.
    for Y_file in Y_files:

        try:
            Y = nib.load(Y_file)
        except Exception as error:
            raise ValueError('The NIFTI "' + Y_file + '"does not exist')

        d = Y.get_data()
        
        # Count number of scans at each voxel
        sumVox = sumVox + 1*(np.nan_to_num(d)!=0)
        print(repr(sumVox))
        print(repr(sum(sum(sum(sumVox)))))

        # Check NIFTI images have the same dimensions.
        if not np.array_equal(d.shape, d0.shape):
            raise ValueError('Input NIFTI "' + Y_file + '" has ' +
                             'different dimensions to "' +
                             Y_files[0] + '"')

        # Check NIFTI images are in the same space.
        if not np.array_equal(Y.affine, Y0aff):
            raise ValueError('Input NIFTI "' + Y_file + '" has a ' +
                             'different affine transformation to "' +
                             Y_files[0] + '"')

    # Get map of number of scans at voxel.
    nsvmap = nib.Nifti1Image(sumVox,
                             Y0.affine,
                             header=Y0.header)
    nib.save(nsvmap, 'blm_vox_nsv.nii')

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use.
    blksize = np.floor(MAXMEM/8/NIFTIsize);

    # Make a temporary directory for batch inputs
    if os.path.isdir('binputs'):
        shutil.rmtree('binputs')
    os.mkdir('binputs')
    # Loop through blocks
    for i in range(0, len(Y_files), int(blksize)):

        # Lower and upper block lengths
        blk_l = i
        blk_u = min(i+int(blksize), len(Y_files))
        index = int(i/int(blksize) + 1)
        
        with open(os.path.join("binputs","Y" + str(index) + ".txt"), "w") as a:

            # List all y for this batch in a file
            for f in Y_files[blk_l:blk_u]:
                a.write(str(f) + os.linesep) 

        np.savetxt(os.path.join("binputs","X" + str(index) + ".csv"), 
                   X[blk_l:blk_u], delimiter=",") 
    
    w.resetwarnings()

if __name__ == "__main__":
    main()
