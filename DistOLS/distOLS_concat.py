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

    # Cycle through batches and add together results.
    for batchNo in range(2,(len(XtX_files)+1)):

        # Sum the batches.
        sumXtX = sumXtX + np.loadtxt(
            os.path.join("binputs","XtX" + str(batchNo) + ".csv"), 
                         delimiter=",")

        sumXtY = sumXtY + np.loadtxt(
            os.path.join("binputs","XtY" + str(batchNo) + ".csv"), 
                         delimiter=",")

    if np.ndim(sumXtX) == 0:
        sumXtX = [sumXtX]

    if np.ndim(sumXtY) == 0:
        sumXtY = [sumXtY]

    beta = np.dot(np.linalg.inv(sumXtX), sumXtY)

    np.savetxt(os.path.join("binputs","beta.csv"), 
                   beta, delimiter=",") 


if __name__ == "__main__":
    main()