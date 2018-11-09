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

    # Get X transpose Y, X transpose X and Y transpose Y.
    XtY = blkXtY(X, Y_files)
    XtX = blkXtX(X)
    YtY = blkYtY(Y_files)

    if len(args)==1:
        # Record XtX and XtY
        np.savetxt(os.path.join("binputs","XtX" + str(batchNo) + ".csv"), 
                   XtX, delimiter=",") 
        np.savetxt(os.path.join("binputs","XtY" + str(batchNo) + ".csv"), 
                   XtY, delimiter=",") 
        np.savetxt(os.path.join("binputs","YtY" + str(batchNo) + ".csv"), 
                   YtY, delimiter=",") 

    else:
        return (XtX, XtY, YtY)


def blkYtY(Y_files)

    # Load in one nifti to check NIFTI size
    Y0 = nib.load(Y_files[0])
    d = Y0.get_data()
    
    # Get number of voxels. ### CAN BE IMPROVED WITH MASK
    nvox = np.prod(d.shape)

    # Number of scans in block
    nscan = len(Y_files)

    # Read in Y
    Y = np.zeros([nvox, 1, nscan])
    Yt = np.zeros([nvox, nscan, 1])

    for i in range(0, len(Y_files)):

        # Read in each individual NIFTI.
        Y_indiv = nib.load(Y_files[i])
        d = Y_indiv.get_data()

        # NaN check
        d = np.nan_to_num(d)

        # Constructing Y matrix
        Y[:, 0, i] = d.reshape([1, nvox])
        Yt[:, i, 0] = d.reshape([1, nvox])

    # Calculate Y transpose Y.
    YtY = np.matmul(Yt,Y)

    return YtY



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

    XtY = np.asarray(
                np.dot(np.transpose(X), Y))

    if np.ndim(XtY) == 0:
        XtY = np.array([[XtY]])
    elif np.ndim(XtY) == 1:
        XtY = np.array([XtY])

    return XtY


def blkXtX(X):

    XtX = np.asarray(
                np.dot(np.transpose(X), X))

    if np.ndim(XtX) == 0:
        XtX = np.array([XtX])
    elif np.ndim(XtX) == 1:
        XtX = np.array([XtX])

    return XtX


if __name__ == "__main__":
    main()
