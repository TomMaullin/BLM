import glob
import numpy as np
import nibabel as nib
import os

def main():

    print('running')
    M_files = glob.glob('/gpfs2/well/nichols/projects/UKB/MNI/*_tfMRI_mask_MNI.nii.gz')
    
    j = 1
    for M_file in M_files:

        print('j: ' + str(j))

        for i in range(10, len(M_files)+1):

            if j <= i:

                print('i: ' + str(i))

                with open(os.path.join(os.getcwd(),'BLM','test','data','ukbb', 'M_files_ukbb_' + str(i) + '.txt'), 'a') as a:

                    a.write(M_file + os.linesep)

            i = i+1

        j = j+1

    print('done')
        

if __name__ == "__main__":
    main()


