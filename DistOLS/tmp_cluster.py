import glob
import nibabel as nib
import os
import distOLS_setup
import numpy as np

def main():

    # Read in the list of NIFTI's needed for this OLS.
    analydir = '/well/nichols/users/kfh142/data'
    os.chdir(analydir)
    Y_files = glob.glob(os.path.join(analydir, "IMAGEN/spmstatsintra/*/SessionB/EPI_short_MID/swea/con_0010.nii"))

    # Design matrix and number of parameters.
    X = np.ones([1815, 2])
    X[1800:, 1] = 0
    X[0:1800, 0] = 0

    distOLS_setup.main(Y_files, X)


if __name__ == "__main__":
    main()
