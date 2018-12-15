import glob
import numpy as np
import nibabel as nib

def main():

    print('running')
    Y_files = glob.glob('/gpfs2/well/nichols/projects/UKB/MNI/*_tfMRI_cope1_MNI.nii.gz')

    for Y_file in Y_files:

        Y = nib.load(Y_file)
        d = Y.get_data()
        d = np.nan_to_num(d)

        d = d[0:d.shape[0],0:d.shape[1],int(np.floor(d.shape[2]/2)):d.shape[2]]


        frac = np.count_nonzero(d)/(d.shape[0]*d.shape[1]*d.shape[2])

        if frac < 0.7:
            print(Y_file)
            print(frac)

if __name__ == "__main__":
    main()


