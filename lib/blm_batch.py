import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
from numpy.lib.format import open_memmap
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import shutil
import yaml
import time
np.set_printoptions(threshold=np.nan)
from lib.fileio import *

def main(*args):


    # Change to blm directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    

    # Obtain batch number
    batchNo = args[0]

    # Work out if which file to look at for inputs
    if len(args)==1 or (not args[1]):
        # Load in inputs
        with open(os.path.join(os.getcwd(),'..','blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream,Loader=yaml.FullLoader)
    else:
        if type(args[1]) is str:
            # In this case inputs file is first argument
            with open(os.path.join(args[1]), 'r') as stream:
                inputs = yaml.load(stream,Loader=yaml.FullLoader)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[1]

    # Work out the maximum memory limit
    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    # Output directory
    OutDir = inputs['outdir']

    # Get number of fixed effects parameters
    L1 = str2vec(inputs['contrasts'][0]['c' + str(1)]['vector'])
    L1 = np.array(L1)
    p = L1.shape[0]
    del L1

    # Y volumes
    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load in one nifti to check NIFTI size
    try:
        Y0 = loadFile(Y_files[0])
    except Exception as error:
        raise ValueError('The NIFTI "' + Y_files[0] + '"does not exist')

    # Read in some data as a default nifti
    d0 = Y0.get_data()

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTImem = sys.getsizeof(np.zeros(d0.shape,dtype='uint64'))

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use.
    blksize = int(np.floor(MAXMEM/8/NIFTImem/p));

    # Reduce X to X for this block.
    X = loadFile(inputs['X'])
    X = X[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]

    # Mask volumes (if they are given)
    if 'data_mask_files' in inputs:

        # Rrad in mask files, making sure to avoid the newline characters
        with open(inputs['data_mask_files']) as a:

            M_files = []
            i = 0
            for line in a.readlines():

                M_files.append(line.replace('\n', ''))

        # If we have a mask for each Y, reduce the list to just for this block
        if len(M_files) == len(Y_files):

            # In this case we have a mask per Y volume
            M_files = M_files[(blksize*(batchNo-1)):min((blksize*batchNo),len(M_files))]

        else:

            # If we haven't the same number of masks and observations then
            # something must be mispecified.
            if len(M_files) > len(Y_files):

                raise ValueError('Too many data_masks specified!')

            else:

                raise ValueError('Too few data_masks specified!')

    # Otherwise we have no masks
    else:

        # There is not a mask for each Y as there are no masks at all!
        M_files = []

    # Mask threshold for Y (if given)
    if 'data_mask_thresh' in inputs:
        M_t = float(inputs['data_mask_thresh'])
    else:
        M_t = None

    # Mask volumes (if they are given)
    if 'analysis_mask' in inputs:

        # Load the file and check it's shape is 3d (as oppose to 4d with a 4th dimension
        # of 1)
        M_a = loadFile(inputs['analysis_mask']).get_data()
        M_a = M_a.reshape((M_a.shape[0],M_a.shape[1],M_a.shape[2]))

    else:

        # Else set to None
        M_a = None

    # Reduce Y_files to only Y files for this block
    Y_files = Y_files[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]
    
    # Verify input
    verifyInput(Y_files, M_files, Y0)

    # Obtain Y, M (essentially the array Y!=0) n_sv and Mmap.
    # This mask is just for voxels with no studies present.
    Y, n_sv, M, Mmap = obtainY(Y_files, M_files, M_t, M_a)

    # Work out voxel specific design
    MX = applyMask(X, M)

    # Get X'Y and Y'Y. 
    # ------------------------------------------------------------------
    # Developer note: For these product matrices we do not need to worry
    # about missing rows in X. This is as the corresponding elements in
    # Y should already be set to 0 and, as such, won't have any affect 
    # on these products.
    # ------------------------------------------------------------------

    # We are careful how we compute X'Y and Y'Y, in case either p is 
    #  large. We save these "chunk by chunk" as memory map objects just
    # in case they don't fit in working memory (this is only usually an
    # issue for very large designs).
    memorySafeAtB(X.reshape(1,X.shape[0],X.shape[1]),Y,MAXMEM,os.path.join(OutDir,"tmp","XtY.npy"))
    memorySafeAtB(Y,Y,MAXMEM,os.path.join(OutDir,"tmp","YtY.npy"))

    # In a spatially varying design XtX has dimensions n by p by p. We
    # reshape to n by p^2 so that we can save as a csv.
    XtX = MX.transpose(0,2,1) @ MX
    XtX = XtX.reshape([XtX.shape[0], XtX.shape[1]*XtX.shape[2]])

    # Record product matrix X'X.
    np.save(os.path.join(OutDir,"tmp","XtX" + str(batchNo)), 
                XtX)

    # Get map of number of observations at voxel.
    n_sv = nib.Nifti1Image(n_sv,
                           Y0.affine,
                           header=Y0.header)
    nib.save(n_sv, os.path.join(OutDir,'tmp',
                    'blm_vox_n_batch'+ str(batchNo) + '.nii'))

    # Get Mmap, indicating which design each voxel must use for analysis,
    # using an integer representing the order in which X'X, Z'X and Z'Z 
    # appear in the `XtX.npy`, `ZtX.npy` and `ZtZ.npy` files respectively.
    Mmap = nib.Nifti1Image(Mmap,
                           Y0.affine,
                           header=Y0.header)
    nib.save(Mmap, os.path.join(OutDir,'tmp',
                    'blm_vox_uniqueM_batch'+ str(batchNo) + '.nii'))

    w.resetwarnings()

def verifyInput(Y_files, M_files, Y0):

    # Obtain information about zero-th scan
    d0 = Y0.get_data()
    Y0aff = Y0.affine

    # Initial checks for NIFTI compatability for Y.
    for i in range(0, len(Y_files)):

        Y_file = Y_files[i]

        try:
            Y = loadFile(Y_file)
        except Exception as error:
            raise ValueError('The NIFTI "' + Y_file + '"does not exist')

        # Check NIFTI images have the same dimensions.
        if not np.array_equal(Y0.shape, Y.shape):
            raise ValueError('Input NIFTI "' + Y_file + '" has ' +
                             'different dimensions to "' +
                             Y0 + '"')

        # Check NIFTI images are in the same space.
        if not np.array_equal(Y.affine, Y0aff):
            raise ValueError('Input NIFTI "' + Y_file + '" has a ' +
                             'different affine transformation to "' +
                             Y0 + '"')

    # Initial checks for NIFTI compatability for M.
    if M_files is not None:
        for i in range(0, len(M_files)):

            M_file = M_files[i]

            try:
                M = loadFile(M_file)
            except Exception as error:
                raise ValueError('The NIFTI "' + M_file + '"does not exist')

            # Check NIFTI images have the same dimensions.
            if not np.array_equal(Y0.shape, M.shape):
                raise ValueError('Input NIFTI "' + M_file + '" has ' +
                                 'different dimensions to "' +
                                 Y0 + '"')

            # Check NIFTI images are in the same space.
            if not np.array_equal(M.affine, Y0aff):
                raise ValueError('Input NIFTI "' + M_file + '" has a ' +
                                 'different affine transformation to "' +
                                 Y0 + '"')


# ============================================================================
# 
# The below function reads in the input files and thresholds and returns; Y
# (as a numpy array), the overall mask (as a 3D numpy array), the spatially
# varying number of observationss (as a 3D numpy array), the array Y!=0 
# (resized appropriately for later computation) and a uniqueness map 
# representing which voxel has which design.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - `Y_files`: A list of input NIFTI volumes.
#  - `M_files`: A list of input NIFTI mask volumes.
#  - `M_t`: A numerical threshold k. Any voxel with less than k input volumes
#           present will be discarded. Can be set to None.
#  - `M_a`: An overall analysis mask 3D numpy array. Can be set to None.
#
# ----------------------------------------------------------------------------
#
# This function gives as outputs:
#
# ----------------------------------------------------------------------------
#
#  - `Y`: The masked observations, reshaped to be of dimension v by n by 1.
#  - `n_sv`: The spatially varying number of observations (as a 3D numpy
#            array).
#  - `M`: The array Y!=0 (resized appropriately for later computation).
#  - `Mmap`: A uniqueness map representing which voxel has which design.
#
# ============================================================================
def obtainY(Y_files, M_files, M_t, M_a):

    # Load in one nifti to check NIFTI size
    Y0 = loadFile(Y_files[0])
    d = Y0.get_data()
    
    # Get number of voxels.
    v = np.prod(d.shape)

    # Number of observations in block
    n = len(Y_files)

    # Count number of observations contributing to voxels
    n_sv = np.zeros(d.shape)

    # Read in Y
    Y = np.zeros([n, v])
    for i in range(0, len(Y_files)):

        # Read in each individual NIFTI.
        Y_indiv = loadFile(Y_files[i])

        # Mask Y if necesary
        if M_files:
        
            # Apply mask
            M_indiv = loadFile(M_files[i]).get_data()
            d = np.multiply(
                Y_indiv.get_data(),
                M_indiv)
        else: 
            #Just load in Y
            d = Y_indiv.get_data()

        # If theres an initial threshold for the data apply it.
        if M_t is not None:
            d[d<M_t]=0

        if M_a is not None:
            d[M_a==0]=0

        # NaN check
        d = np.nan_to_num(d)

        # Count number of observations at each voxel
        n_sv = n_sv + 1*(np.nan_to_num(d)!=0)

        # Constructing Y array
        Y[i, :] = d.reshape([1, v])
    
    # Work out mask
    Mask = np.zeros([v])
    Mask[np.where(np.count_nonzero(Y, axis=0)>0)[0]] = 1
    
    # Apply full mask to Y
    Y_fm = Y[:, np.where(np.count_nonzero(Y, axis=0)>0)[0]]

    # Apply analysis mask to Y, we use the analysis mask here as the product
    # matrices across all batches should have the same masking for convinience
    # We can apply the full mask at a later stage.
    Y = Y[:, np.where(M_a.reshape([v]))[0]]

    # Work out the mask.
    M = (Y_fm!=0)

    # Get indices corresponding to the unique rows of M
    M_df = pd.DataFrame(M.transpose())
    M_df['id'] = M_df.groupby(M_df.columns.tolist(), sort=False).ngroup() + 1
    unique_id_nifti = M_df['id'].values

    # Make a nifti which will act as a "key" telling us which voxel had which design
    Mmap = np.zeros(Mask.shape)
    Mmap[np.flatnonzero(Mask)] = unique_id_nifti[:]
    Mmap = Mmap.reshape(n_sv.shape)

    # Get the unique columns of M (Care must be taken here to retain
    # the original ordering, as unique_id_nifti is now based on said
    # ordering)
    _, idx = np.unique(M, axis=1, return_index=True)
    M = M[:,np.sort(idx)]

    # Reshape Y
    Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))

    # Return results
    return Y, n_sv, M, Mmap


# ============================================================================
# 
# The below function takes in a (2D) array, X, and applies a mask to it, 
# resulting in a 3D array, MX, where whenever data was missing at voxel v for
# observations i1, i2,... etc, MX[v,:,:] is X but with rows i1, i2,... i3
# replaced with zeros.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - `X`: The (2D) array of interest to be converted.
#  - `M`: The (n by v) mask to be applied to X (essentially the array Y!=0,
#         reshaped where appropriate).
#
# ----------------------------------------------------------------------------
#
# This function gives as outputs:
#
# ----------------------------------------------------------------------------
#
#  - `MX`: The (3D) "Masked" version of X.
#
# ============================================================================
def applyMask(X,M):

    # Get M in a form where each voxel's mask is mutliplied
    # by X
    M = M.transpose().reshape([M.shape[1], 1, M.shape[0]])
    Xt=X.transpose()

    # Obtain X for each voxel
    MXt = np.multiply(M, Xt)
    MX = MXt.transpose(0,2,1)

    return MX


# ============================================================================
#
# Given two 3D numpy arrays, A and B, of shape (1, k1, k2) and (v, k1, k3)
# respectively, the below function calculates the (v, k2, k3) matrix A'B
# and outputs it to a file in a "memory safe" way, ensuring that the 
# (v, k2, k3) matrix is calculated and output in managable chunks, one at a
# time.
#
# This function is designed with the use case of the product matrices X'Y and
# Z'Y in mind.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `A`: The (1, k1, k2) shaped matrix.
# - `B`: The (v, k1, k3) shaped matrix.
# - `MAXMEM`: The maximum memory allowed for usage, in bytes.
# - `filename`: The name of the file.
#
# ============================================================================
def memorySafeAtB(A,B,MAXMEM,filename):

    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            os.open(filename + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True

    # Record v and k3 (which is usually p or 1)
    v = B.shape[0]
    pORone = A.shape[2]

    # If the memory map doesn't exist already, create it
    if not os.path.isfile(filename):

        # Create a memory-mapped .npy file with the dimensions and dtype we want
        M = open_memmap(filename, mode='w+', dtype='float64', shape=(v,pORone))

        # Work out the number of voxels we can save at a time.
        # (8 bytes per numpy float exponent multiplied by 10
        # for a safe overhead)
        vPerBlock = MAXMEM/(10*8*pORone)

        # Work out the indices for each group of voxels
        voxelGroups = np.array_split(np.arange(v, dtype='int32'), v//vPerBlock+1)

        # Loop through each group of voxels saving A'B for those voxels
        for vb in range(int(v//vPerBlock+1)):
            M[voxelGroups[vb],:]=(A.transpose(0,2,1) @ B[voxelGroups[vb],:,:]).reshape(len(voxelGroups[vb]),pORone)
    
    # Otherwise we add to the memory map that does exist
    else:

        # Load in the file but in memory map mode
        M = np.load(filename,mmap_mode='r+')
        M = M.reshape((v,pORone))

        # Work out the number of voxels we can save at a time.
        # (8 bytes per numpy float exponent multiplied by 10
        # for a safe overhead)
        vPerBlock = MAXMEM/(10*8*pORone)

        # Work out the indices for each group of voxels
        voxelGroups = np.array_split(np.arange(v, dtype='int32'), v//vPerBlock+1)
        
        # Loop through each group of voxels saving A'B for those voxels
        for vb in range(int(v//vPerBlock+1)):
            M[voxelGroups[vb],:]=M[voxelGroups[vb],:]+(A.transpose(0,2,1) @ B[voxelGroups[vb],:,:]).reshape(len(voxelGroups[vb]),pORone)

    # Delete M from memory (important!)
    del M

    # Delete lock file, so other jobs know they can now write to the
    # file
    os.remove(filename + ".lock")

if __name__ == "__main__":
    main()
