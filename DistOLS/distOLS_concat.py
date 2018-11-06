import numpy as np
import subprocess
import warnings
import resource
from lib import blkXtX, blkXtY
import nibabel as nib
import sys
import os
import shutil

def main():

    # Change to distOLS directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Work out which files we need.
    XtX_files = glob.glob("XtX*")
    XtY_files = glob.glob("XtY*")

    # Read the matrices from the first batch.
    sumXtX = np.loadtxt(os.path.join("binputs","XtX0.csv"), 
                        delimiter=",")
    sumXtY = np.loadtxt(os.path.join("binputs","XtY0.csv"), 
                        delimiter=",")

    # Cycle through batches and add together results.
    for batchNo in range(1,len(XtX_files)):

        # Sum the batches.
        sumXtX = sumXtX + np.loadtxt(
            os.path.join("binputs","XtX" + str(batchNo) + ".csv"), 
                         delimiter=",")

        sumXtY = sumXtY + np.loadtxt(
            os.path.join("binputs","XtY" + str(batchNo) + ".csv"), 
                         delimiter=",")

    beta = np.dot(numpy.linalg.inv(sumXtX), XtY)

    np.savetxt(os.path.join("binputs","beta.csv"), 
                   beta, delimiter=",") 


if __name__ == "__main__":
    main()