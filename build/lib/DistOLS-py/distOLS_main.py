import sys
import scipy.ndimage
import nibabel as nib
import numpy as np
import subprocess
import warnings
import resource
from lib import blkXtX, blkXtY

def main(Y_files, X): #flag='spat'

    # Maximum Memory - currently using SPM default of 2**29
    MAXMEM = 2**29

    # Load in one nifti to check NIFTI size
    Y0 = nib.load(Y_files[0])
    d = Y0.get_data()

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTIsize = sys.getsizeof(np.zeros(d.shape,dtype='uint64'))

    # Get number of voxels. ### CAN BE IMPROVED WITH MASK
    nvox = np.prod(d.shape)

    # Number of parameters
    npar = X.shape[1]

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use.
    blksize = np.floor(MAXMEM/8/NIFTIsize);

    # Initialise empty sum matrices.
    sumXtX = np.zeros([npar, npar])
    sumXtY = np.zeros([npar, nvox])

    # Loop through blocks
    for i in range(0, len(Y_files), int(blksize)):

        # Lower and upper block lengths
        blk_l = i*int(blksize)
        blk_u = min((i+1)*int(blksize), len(Y_files))

        # Work out X transpose X and X transpose Y for this block
        # and add to running total.
        sumXtX = sumXtX + blkXtX.blkXtX(X[blk_l:blk_u,:]);
        sumXtY = sumXtY + blkXtY.blkXtY(X[blk_l:blk_u,:], Y_files[blk_l:blk_u])

    # Calculate blocks
    beta = sumXtX @ sumXtY

    # Save the result.
    for i in range(0, beta.shape[0]):

        # Make beta i Nifti object
        betai = nib.Nifti1Image(np.reshape(beta[i,:], d.shape),
                                Y0.affine,
                                header=Y0.header)

        # Save beta i Nifti object
        nib.save(betai, 'beta1.nii.gz')

if __name__ == "__main__":
    main()