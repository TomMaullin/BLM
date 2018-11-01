import glob
import nibabel as nib
import os
import distOLS_setup
import numpy as np

def main():

    # Read in the list of NIFTI's needed for this OLS.
    analydir = '/home/tommaullin/Documents/SwE-Toolbox-TestData/test_p_t_img/ground_truth';
    os.chdir(analydir);
    Y_files = glob.glob("*")

    # Design matrix and number of parameters.
    X = np.ones([1815, 1])

    distOLS_setup.main(Y_files, X)


if __name__ == "__main__":
    main()
