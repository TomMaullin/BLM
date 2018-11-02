import numpy as np
import subprocess
import warnings
import resource
from lib import blkXtX, blkXtY
import nibabel as nib
import sys
import os
import shutil

def main(*args):

    # In the batch mode we are given a batch number pointing us to
    # the correct files
    if len(args)==1:

        batchNo = args[0];

        with open(os.path.join("binputs","Y" + str(batchNo) + ".txt")) as a:

            Y_files = []
            i = 0
            for line in a.readlines():

                print(repr(line))

                Y_files.append(line.replace('\n', ''))

        X = np.loadtxt(os.path.join("binputs","X" + str(batchNo) + ".csv"), 
                       delimiter=",") 

        print(repr(X))
        print(repr(Y_files))





if __name__ == "__main__":
    main()
