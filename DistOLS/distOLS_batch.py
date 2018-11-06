import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import shutil

def main(*args):

    # Change to distOLS directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    

    # In the batch mode we are given a batch number pointing us to
    # the correct files
    if len(args)==1:

        batchNo = args[0];
        with open(os.path.join("binputs","Y" + str(batchNo) + ".txt")) as a:

            Y_files = []
            i = 0
            for line in a.readlines():

                Y_files.append(line.replace('\n', ''))

        X = np.loadtxt(os.path.join("binputs","X" + str(batchNo) + ".csv"), 
                       delimiter=",") 
    
    else:

        Y_files = arg[0]
        X = arg[1]

    # Get X transpose Y and X transpose X
    XtY = blkXtY(X, Y_files)
    XtX = blkXtX(X)

    if len(args)==1:
        # Record XtX and XtY
        print(repr(XtX))
        np.savetxt(os.path.join("binputs","XtX" + str(batchNo) + ".csv"), 
                   XtX, delimiter=",") 
        np.savetxt(os.path.join("binputs","XtY" + str(batchNo) + ".csv"), 
                   XtY, delimiter=",") 

    else:
        return (XtX, XtY)


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

    return np.dot(np.transpose(X), Y)


def blkXtX(X):

    return np.dot(np.transpose(X), X)

if __name__ == "__main__":
    main()
