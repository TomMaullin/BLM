from scipy import ndimage
import pandas as pd
import time
from blm.lib.fileio import *
import yaml
import shutil
import glob
import os
import sys
import nibabel as nib
import numpy as np
import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action='ignore', category=FutureWarning)


def generate_data(n, p, dim, OutDir, simNo):
    """
    Generates simulated data with the specified dimensions and other parameters.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of parameters.
    dim : numpy array
        Dimensions of data to be generated. Must be given as a numpy array.
    OutDir : str
        Output directory path where the generated data will be saved.
    simNo : int
        Simulation number for generating the data.

    Returns
    -------
    None
        This function does not return any value. It saves the generated data in the specified output directory.
    """

    # Get simulation directory
    simDir = os.path.join(OutDir, 'sim' + str(simNo))

    # Make sure in numpy format (added 20 for smoothing)
    origdim = np.array(dim)
    dim = origdim + 20

    # -------------------------------------------------
    # Design parameters
    # -------------------------------------------------

    # fwhm for smoothing
    fwhm = 5

    # Number of voxels
    v = np.prod(dim)

    # Relative missingness threshold (percentage)
    rmThresh = 0.5

    # -------------------------------------------------
    # Obtain design matrix
    # -------------------------------------------------

    # Fixed effects design matrix
    X = get_X(n, p)

    # -------------------------------------------------
    # Obtain beta parameter vector
    # -------------------------------------------------

    # Get beta
    beta = get_beta(p)

    # -----------------------------------------------------
    # Obtain Y
    # -----------------------------------------------------

    # Work out Xbeta
    Xbeta = X @ beta

    # Loop through subjects generating nifti images
    for i in np.arange(n):

        # Initialize Yi to Xi times beta
        Yi = Xbeta[0, i, 0]

        # Get epsiloni
        epsiloni = get_epsilon(v, 1).reshape(dim)

        # Add epsilon to Yi
        Yi = Yi + epsiloni

        # Smooth Y_i
        Yi = smooth_data(Yi, 3, [fwhm]*3, trunc=6,
                         scaling='kernel').reshape(dim)

        # Obtain mask
        mask = get_random_mask(dim).reshape(Yi.shape)

        # Mask Yi
        Yi = Yi*mask

        # Truncate off (handles smoothing edge effects)
        Yi = Yi[10:(dim[0]-10), 10:(dim[1]-10), 10:(dim[2]-10)]

        # Output Yi
        addBlockToNifti(os.path.join(simDir, "data", "Y"+str(i)+".nii"),
                        Yi, np.arange(np.prod(origdim)), volInd=0, dim=origdim)

    # -----------------------------------------------------
    # Save X
    # -----------------------------------------------------

    # Write out Z in full to a csv file
    pd.DataFrame(X.reshape(n, p)).to_csv(os.path.join(
        simDir, "data", "X.csv"), header=None, index=None)

    # -----------------------------------------------------
    # Contrast vector
    # -----------------------------------------------------

    # Make a simple string representing the contrast vector to test
    contrast_vec = '['
    for i in range(p-1):
        contrast_vec = contrast_vec + '0, '
    contrast_vec = contrast_vec + '1]'

    # -----------------------------------------------------
    # Inputs file
    # -----------------------------------------------------

    # Write to an inputs file
    with open(os.path.join(simDir, 'inputs.yml'), 'a') as f:

        # X, Y, Z and Masks
        f.write("Y_files: " + os.path.join(simDir,
                "data", "Yfiles.txt") + os.linesep)
        f.write("X: " + os.path.join(simDir, "data", "X.csv") + os.linesep)

        # Output directory
        f.write("outdir: " + os.path.join(simDir, "BLM") + os.linesep)

        # Missingness percentage
        f.write("Missingness: " + os.linesep)
        f.write("  MinPercent: " + str(rmThresh) + os.linesep)

        # Let's not output covariance maps for now!
        f.write("OutputCovB: False" + os.linesep)

        # Contrast vectors
        f.write("contrasts: " + os.linesep)
        f.write("  - c1: " + os.linesep)
        f.write("      name: null_contrast" + os.linesep)
        f.write("      vector: " + contrast_vec + os.linesep)
        f.write("      statType: T " + os.linesep)

        # Voxel-wise batching for speedup - not necessary - but
        # convenient
        f.write("voxelBatching: 1" + os.linesep)
        f.write("MAXMEM: 2**34" + os.linesep)

        # Cluster parameters
        f.write("clusterType: SLURM" + os.linesep)
        f.write("numNodes: 100" + os.linesep)

        # Log directory and simulation mode (backdoor options)
        f.write("sim: 1" + os.linesep)
        f.write("logdir: " + os.path.join(simDir, "simlog") + os.linesep)

    # -----------------------------------------------------
    # Yfiles.txt
    # -----------------------------------------------------
    with open(os.path.join(simDir, "data", 'Yfiles.txt'), 'a') as f:

        # Loop through listing mask files in text file
        for i in np.arange(n):

            # Write filename to text file
            if i < n-1:
                f.write(os.path.join(simDir, "data",
                        "Y"+str(i)+".nii") + os.linesep)
            else:
                f.write(os.path.join(simDir, "data", "Y"+str(i)+".nii"))

    # -----------------------------------------------------
    # Version of data which can be fed into R
    # -----------------------------------------------------
    #  - i.e. seperate Y out into thousands of csv files
    #         each containing number of subjects by 1000
    #         voxel arrays.
    # -----------------------------------------------------

    # Number of voxels in each batch
    nvb = 1000

    # Work out number of groups we have to split indices into.
    nvg = int(np.prod(origdim)//nvb)

    # Write out the number of voxel groups we split the data into
    with open(os.path.join(simDir, "data", "nb.txt"), 'w') as f:
        print(int(nvg), file=f)


# R preprocessing
def Rpreproc(OutDir, simNo, dim, nvg, cv):

    # Get simulation directory
    simDir = os.path.join(OutDir, 'sim' + str(simNo))

    # Make sure in numpy format
    dim = np.array(dim)

    # Number of voxels
    v = np.prod(dim)

    # There should be an inputs file in each simulation directory
    with open(os.path.join(simDir, 'inputs.yml'), 'r') as stream:
        inputs = yaml.load(stream, Loader=yaml.FullLoader)

    # Number of observations
    X = pd.io.parsers.read_csv(os.path.join(
        simDir, "data", "X.csv"), header=None).values
    n = X.shape[0]

    # Relative masking threshold
    rmThresh = inputs['Missingness']['MinPercent']

    # Split voxels we want to look at into groups we can compute
    voxelGroups = np.array_split(np.arange(v), nvg)

    # Current group of voxels
    inds_cv = voxelGroups[cv]

    # Number of voxels currently (should be ~1000)
    v_current = len(inds_cv)

    # Loop through each subject reading in Y and reducing to just the voxels
    # needed
    for i in np.arange(n):

        # Load in the Y volume
        Yi = nib.load(os.path.join(simDir, "data",
                      "Y"+str(i)+".nii")).get_fdata()

        # Flatten Yi
        Yi = Yi.reshape(v)

        # Get just the voxels we're interested in
        Yi = Yi[inds_cv].reshape(1, v_current)

        # Concatenate
        if i == 0:
            Y_concat = Yi
        else:
            Y_concat = np.concatenate((Y_concat, Yi), axis=0)

    # Loop through voxels checking missingness
    for vox in np.arange(v_current):

        # Threshold out the voxels which have too much missingness
        if np.count_nonzero(Y_concat[:, vox], axis=0)/n < rmThresh:

            # If we don't have enough data lets replace that voxel
            # with zeros
            Y_concat[:, vox] = np.zeros(Y_concat[:, vox].shape)

    # Write out Z in full to a csv file
    pd.DataFrame(Y_concat.reshape(n, v_current)).to_csv(os.path.join(
        simDir, "data", "Y_Rversion_" + str(cv) + ".csv"), header=None, index=None)


def get_random_mask(dim):

    # FWHM
    fwhm = 10

    # Load analysis mask
    mu = nib.load(os.path.join(os.path.dirname(
        __file__), 'mask.nii')).get_fdata()

    # Add some noise and smooth
    mu = smooth_data(mu + 8*np.random.randn(*(mu.shape)), 3, [fwhm]*3)

    # Re-threshold (this has induced a bit of randomness in the mask shape)
    mu = 1*(mu > 0.6)

    return (mu)


def get_X(n, p):

    # Generate random X.
    X = np.random.uniform(low=-0.5, high=0.5, size=(n, p))

    # Make the first column an intercept
    X[:, 0] = 1

    # Reshape to dimensions for broadcasting
    X = X.reshape(1, n, p)

    # Return X
    return (X)


def get_beta(p):

    # Make beta (we just have beta = [p-1,p-2,...,0])
    beta = p-1-np.arange(p)

    # Reshape to dimensions for broadcasting
    beta = beta.reshape(1, p, 1)

    # Return beta
    return (beta)


def get_sigma2(v):

    # Make sigma2 (for now just set to one across all voxels)
    sigma2 = 10  # np.ones(v).reshape(v,1)

    # Return sigma
    return (sigma2)


def get_epsilon(v, n):

    # Get sigma2
    sigma2 = get_sigma2(v)

    # Make epsilon.
    epsilon = sigma2*np.random.randn(v, n)

    # Reshape to dimensions for broadcasting
    epsilon = epsilon.reshape(v, n, 1)

    return (epsilon)


def get_Y(X, beta, epsilon):

    # Generate the response vector
    Y = X @ beta + epsilon

    # Return Y
    return (Y)


# ============================================================================
#
# The below function adds a block of voxels to a pre-existing NIFTI or creates
# a NIFTI of specified dimensions if not.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `fname`: An absolute path to the Nifti file.
# - `block`: The block of values to write to the NIFTI.
# - `blockInds`: The indices representing the 3D coordinates `block` should be
#                written to in the NIFTI. (Note: It is assumed if the NIFTI is
#                4D we assume that the indices we want to write to in each 3D
#                volume/slice are the same across all 3D volumes/slices).
# - `dim` (optional): If creating the NIFTI image for the first time, the
#                     dimensions of the NIFTI image must be specified.
# - `volInd` (optional): If we only want to write to one 3D volume/slice,
#                        within a 4D file, this specifies the index of the
#                        volume of interest.
# - `aff` (optional): If creating the NIFTI image for the first time, the
#                     affine of the NIFTI image must be specified.
# - `hdr` (optional): If creating the NIFTI image for the first time, the
#                     header of the NIFTI image must be specified.
#
# ============================================================================
def addBlockToNifti(fname, block, blockInds, dim=None, volInd=None, aff=None, hdr=None):

    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            f = os.open(fname + ".lock", os.O_CREAT | os.O_EXCL | os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True

    # Check volInd is correct datatype
    if volInd is not None:

        volInd = int(volInd)

    # Check whether the NIFTI exists already
    if os.path.isfile(fname):

        # Work out dim if we don't already have it
        dim = nib.Nifti1Image.from_filename(fname, mmap=False).shape

        # Work out data
        data = nib.Nifti1Image.from_filename(
            fname, mmap=False).get_fdata().copy()

        # Work out affine
        affine = nib.Nifti1Image.from_filename(fname, mmap=False).affine.copy()

    else:

        # If we know how, make the NIFTI
        if dim is not None:

            # Make data
            data = np.zeros(dim)

            # Make affine
            if aff is None:
                affine = np.eye(4)
            else:
                affine = aff

        else:

            # Throw an error because we don't know what to do
            raise Exception('NIFTI does not exist and dimensions not given')

    # Work out the number of output volumes inside the nifti
    if len(dim) == 3:

        # We only have one volume in this case
        n_vol = 1
        dim = np.array([dim[0], dim[1], dim[2], 1])

    else:

        # The number of volumes is the last dimension
        n_vol = dim[3]

    # Seperate copy of data for outputting
    data_out = np.array(data).reshape(dim)

    # Work out the number of voxels
    n_vox = np.prod(dim[:3])

    # Reshape
    data = data.reshape([n_vox, n_vol])

    # Add all the volumes
    if volInd is None:

        # Add block
        data[blockInds, :] = block.reshape(data[blockInds, :].shape)

        # Cycle through volumes, reshaping.
        for k in range(0, data.shape[1]):

            data_out[:, :, :, k] = data[:, k].reshape(int(dim[0]),
                                                      int(dim[1]),
                                                      int(dim[2]))

    # Add the one volume in the correct place
    else:

        # We're only looking at this volume
        data = data[:, volInd].reshape((n_vox, 1))

        # Add block
        data[blockInds, :] = block.reshape(data[blockInds, :].shape)

        # Put in the volume
        data_out[:, :, :, volInd] = data[:, 0].reshape(int(dim[0]),
                                                       int(dim[1]),
                                                       int(dim[2]))

    # Save NIFTI
    nib.save(nib.Nifti1Image(data_out, affine, header=hdr), fname)

    # Delete lock file, so other jobs know they can now write to the
    # file
    os.remove(fname + ".lock")
    os.close(f)

    del fname, data_out, affine, data, dim


# Smoothing function
def smooth_data(data, D, fwhm, trunc=6, scaling='kernel'):

    # -----------------------------------------------------------------------
    # Reformat fwhm
    # -----------------------------------------------------------------------

    # Format fwhm and replace None with 0
    fwhm = np.asarray([fwhm]).ravel()
    fwhm = np.asarray([0. if elem is None else elem for elem in fwhm])

    # Non-zero dimensions
    D_nz = np.sum(fwhm > 0)

    # Convert fwhm to sigma values
    sigma = fwhm / np.sqrt(8 * np.log(2))

    # -----------------------------------------------------------------------
    # Perform smoothing (this code is based on `_smooth_array` from the
    # nilearn package)
    # -----------------------------------------------------------------------

    # Loop through each dimension and smooth
    for n, s in enumerate(sigma):

        # If s is non-zero smooth by s in that direction.
        if s > 0.0:

            # Perform smoothing in nth dimension
            ndimage.gaussian_filter1d(
                data, s, output=data, mode='constant', axis=n, truncate=trunc)

    # -----------------------------------------------------------------------
    # Rescale
    # -----------------------------------------------------------------------
    if scaling == 'kernel':

        # -----------------------------------------------------------------------
        # Rescale smoothed data to standard deviation 1 (this code is based on
        # `_gaussian_kernel1d` from the `scipy.ndimage` package).
        # -----------------------------------------------------------------------

        # Calculate sigma^2
        sigma2 = sigma*sigma

        # Calculate kernel radii
        radii = np.int16(trunc*sigma + 0.5)

        # Initialize array for phi values (has to be object as dimensions can
        # vary in length)
        phis = np.empty(shape=(D_nz), dtype=object)

        # Index for non-zero dimensions
        j = 0

        # Loop through dimensions to get scaling constants
        for k in np.arange(D):

            # Skip the non-smoothed dimensions
            if fwhm[k] != 0:

                # Get range of values for this dimension
                r = np.arange(-radii[k], radii[k]+1)

                # Get the kernel for this dimension
                phi = np.exp(-0.5 / sigma2[k] * r ** 2)

                # Normalise phi
                phi = phi / phi.sum()

                # Add phi to dictionary
                phis[j] = phi[::-1]

                # Increment j
                j = j + 1

        # Create the D_nz dimensional grid
        grids = np.meshgrid(*phis)

        # Initialize normalizing constant
        ss = 1

        # Loop through axes and take products
        for j in np.arange(D_nz-1):

            # Smoothing kernel along plane (j,j+1)
            product_gridj = (grids[j]*(grids[j+1]*np.ones(grids[0].shape)).T)

            # Get normalizing constant along this plane
            ssj = np.sum((product_gridj)**2)

            # Add to running smoothing constant the sum of squares of this kernel
            # (Developer note: This is the normalizing constant. When you smooth
            # you are mutliplying everything by a grid of values along each dimension.
            # To restandardize you then need to take the sum the squares of this grid
            # and squareroot it. You then divide your data by this number at the end.
            # This must be done once for every dimension, hence the below product.)
            ss = ssj*ss

        # Rescale noise
        data = data/np.sqrt(ss)

    elif scaling == 'max':

        # Rescale noise by dividing by maximum value
        data = data/np.max(data)

    return (data)
