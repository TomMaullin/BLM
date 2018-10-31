# This function takes in the X matrix for a given block and a 
# list of Y_files and returns X transpose Y
# ==============================================================
# USAGE: XtX = blkXtX(X)
# --------------------------------------------------------------
# It takes the following inputs:
#   X - the design matrix.
#   Y_files - A list of files containing scans for regression.
# ==============================================================
# Author: Tom Maullin
import numpy as np
import nibabel as nib

def blkXtY(X, Y_files):

    # Load in one nifti to check NIFTI size
    Y0 = nib.load(Y_files[0])
    d = Y0.get_data()
    
    # Get number of voxels. ### CAN BE IMPROVED WITH MASK
    nvox = np.prod(d.shape)

    # Number of scans in block
    nscan = len(Y_files)

    # Read in Y
    Y = np.zeros([nscan, nvox])
    for i in range(0, len(Y_files)):

        # Read in each individual NIFTI.
        Y_indiv = nib.load(Y_files[i])
        d = Y_indiv.get_data()

        # NaN check
        d = np.nan_to_num(d)

        # Constructing Y matrix
        Y[i, :] = d.reshape([1, nvox])

    return np.transpose(X) @ Y