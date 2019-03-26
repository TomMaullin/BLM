
import os
import nibabel as nib

# This is a small function to load in a file based on it's prefix.
def blm_load(filepath):

    # If the file exists load it.
    try:
        nifti = nib.load(filepath)
    except:
        try:
            if os.path.isfile(os.path.join(filepath, '.nii.gz')):
                nifti = nib.load(os.path.join(filepath, '.nii.gz'))
            elif os.path.isfile(os.path.join(filepath, '.nii')):
                nifti = nib.load(os.path.join(filepath, '.nii'))
            elif os.path.isfile(os.path.join(filepath, '.img.gz')):
                nifti = nib.load(os.path.join(filepath, '.img.gz'))
            else:
                nifti = nib.load(os.path.join(filepath, '.img'))
        except:
            raise ValueError('Input file not found: ' + str(filepath))

    return nifti

