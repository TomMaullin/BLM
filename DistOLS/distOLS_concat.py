import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import glob
import shutil

def main():

    # Change to distOLS directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Work out which files we need.
    XtX_files = glob.glob("XtX*")
    XtY_files = glob.glob("XtY*")

    # Read the matrices from the first batch.
    sumXtX = np.loadtxt(os.path.join("binputs","XtX1.csv"), 
                        delimiter=",")
    sumXtY = np.loadtxt(os.path.join("binputs","XtY1.csv"), 
                        delimiter=",")

    print(sumXtY.ndim)

    # Cycle through batches and add together results.
    for batchNo in range(2,(len(XtX_files)+1)):

        # Sum the batches.
        sumXtX = sumXtX + np.loadtxt(
            os.path.join("binputs","XtX" + str(batchNo) + ".csv"), 
                         delimiter=",")

        sumXtY = sumXtY + np.loadtxt(
            os.path.join("binputs","XtY" + str(batchNo) + ".csv"), 
                         delimiter=",")

    # Dimension bug handling
    if np.ndim(sumXtX) == 0:
        sumXtX = np.array([[sumXtX]])
    elif np.ndim(sumXtX) == 1:
        sumXtX = np.array([sumXtX])

    if np.ndim(sumXtY) == 0:
        sumXtY = np.array([[sumXtY]])
    elif np.ndim(sumXtY) == 1:
        sumXtY = np.array([sumXtY])

    # np linalg inverse doesn't handle dim=[1,1]
    if np.ndim(sumXtX) == 1:
        isumXtX = 1/sumXtX
    else:
        isumXtX = np.linalg.inv(sumXtX)

    # Read in the nifti size.
    NIFTIsize = np.loadtxt(os.path.join("binputs","NIFTIsize.csv"), 
                        delimiter=",")

    beta = np.dot(isumXtX, sumXtY)

    # TODO: HANDLE MULTI BETA Dimensions
    beta1 = beta.reshape(NIFTIsize[0],
                         NIFTIsize[1],
                         NIFTIsize[2])

    # tmp code to output nifti
    nifti = nib.load('IMAGEN/spmstatsintra/000070830069/SessionB/EPI_short_MID/swea/con_0010.nii')

    beta1map = nib.Nifti1Image(beta1,
                               nifti.affine,
                               header=nifti.header)
    nib.save(beta1map, 'tmp.nii')


    np.savetxt(os.path.join("binputs","beta.csv"), 
                   beta1, delimiter=",") 


if __name__ == "__main__":
    main()