import glob
import numpy as np
import nibabel as nib

Y_files = glob.glob('gpfs2/well/nichols/projects/UKB/MNI/*_tfMRI_cope1_MNI.nii.gz')

for Y_file in Y_files:

    print(Y_file)

    Y = nib.load(Y_file)
    d = Y.get_data()
    d = np.nan_to_num(d)

    frac = np.count_nonzero(d)/(d.shape[0]*d.shape[1]*d.shape[2])

        print(frac)

    if frac > 0.3:
        print(Y_file)


