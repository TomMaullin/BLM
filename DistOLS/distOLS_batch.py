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
from DistOLS import distOLS_defaults

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

        
        # Check if we are doing spatially varying.
        inputs = distOLS_defaults.main()
        SVFlag = inputs[3]
        del inputs
        
    else:

        Y_files = args[0]
        X = args[1]

        if len(args)==3:
            SVFlag = args[3]
        else:
            inputs = distOLS_defaults.main()
            SVFlag = inputs[3]
            del inputs

    # WIP PLAN: for spatially varying,
    #if SVFlag:
    #    X = MX(X, Y_files)

    # Obtain Y and a mask for Y. This mask is just for voxels
    # with no studies present.
    Y, Mask = obtainY(Y_files)
    print('Y')
    print(Y)
    print('Mask')
    print(Mask)

    # Get X transpose Y, X transpose X and Y transpose Y.
    XtY = blkXtY(X, Y, Mask)
    XtX = blkXtX(X)
    YtY = blkYtY(Y, Mask)

    print('XtY shape')
    print(XtY.shape)
    print('YtY shape')
    print(YtY.shape)

    if len(args)==1:
        # Record XtX and XtY
        np.savetxt(os.path.join("binputs","XtX" + str(batchNo) + ".csv"), 
                   XtX, delimiter=",") 
        np.savetxt(os.path.join("binputs","XtY" + str(batchNo) + ".csv"), 
                   XtY, delimiter=",") 
        np.savetxt(os.path.join("binputs","YtY" + str(batchNo) + ".csv"), 
                   YtY, delimiter=",") 
        w.resetwarnings()

    else:
        w.resetwarnings()
        return (XtX, XtY, YtY)

def obtainY(Y_files):

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

    print(np.where(np.count_nonzero(Y, axis=0)>1)[0])
    print(np.where(np.count_nonzero(Y, axis=0)>1)[0].shape)
    
    Mask = np.zeros([nvox])
    Mask[np.where(np.count_nonzero(Y, axis=0)>1)[0]] = 1

    Y = Y[:, np.where(np.count_nonzero(Y, axis=0)>1)[0]]

    return Y, Mask

# Note: this techniqcally calculates sum(Y.Y) for each voxel,
# not Y transpose Y for all voxels
def blkYtY(Y, Mask):

    # Read in number of scans and voxels.
    nscan = Y.shape[0]
    nvox = Y.shape[1]

    # Reshape Y
    Y_rs = Y.transpose().reshape(nvox, nscan, 1)
    Yt_rs = Y.transpose().reshape(nvox, 1, nscan)
    del Y

    # Calculate Y transpose Y.
    YtY_m = np.matmul(Yt_rs,Y_rs).reshape([nvox, 1])

    # Unmask YtY
    YtY = np.zeros([YtY_m.shape[0], 1])
    YtY[np.where(Mask.any(axis=0))[0],:] = YtY_m[:]

    print('YtYYYY')
    print(YtY)

    return YtY


def blkXtY(X, Y, Mask):
    
    # Calculate X transpose Y
    XtY = np.asarray(
                np.dot(np.transpose(X), Y))

    # Check the dimensions haven't been reduced
    # (numpy will lower the dimension of the 
    # array if the length in one dimension is
    # one)
    if np.ndim(XtY) == 0:
        XtY = np.array([[XtY]])
    elif np.ndim(XtY) == 1:
        XtY = np.array([XtY])

    return XtY


def blkXtX(X):

    # Calculate XtX
    XtX = np.asarray(
                np.dot(np.transpose(X), X))

    # Check the dimensions haven't been reduced
    # (numpy will lower the dimension of the 
    # array if the length in one dimension is
    # one)
    if np.ndim(XtX) == 0:
        XtX = np.array([XtX])
    elif np.ndim(XtX) == 1:
        XtX = np.array([XtX])

    return XtX


if __name__ == "__main__":
    main()
