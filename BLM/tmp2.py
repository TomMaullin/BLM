import glob
import numpy as np
import nibabel as nib
import os

def main():

    print('running')
    Y_files = glob.glob('/gpfs2/well/nichols/projects/UKB/MNI/*_tfMRI_cope1_MNI.nii.gz')

    j = 1
    for Y_file in Y_files:

    	if j >= 10:

            print(str(j))
            X = np.ones([1,j])
            np.savetxt(os.path.join(os.getcwd(),'BLM','test','data','ukbb', 'X_ukbb_' + str(j) + '.csv'), X, delimiter=",")

    	j = j + 1

    print('done')        

if __name__ == "__main__":
    main()

