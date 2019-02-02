import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import shutil
import yaml
import pandas
import time
np.set_printoptions(threshold=np.nan)

def main(*args):

    # Change to blm directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    

    batchNo = args[0]

    if len(args)==1:
        # Load in inputs
        with open(os.path.join(os.getcwd(),'..','blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        # In this case inputs is first argument
        inputs = args[1]

    MAXMEM = eval(inputs['MAXMEM'])    
    OutDir = inputs['outdir']

    # Y volumes
    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load in one nifti to check NIFTI size
    try:
        Y0 = nib.load(Y_files[0])
    except Exception as error:
        raise ValueError('The NIFTI "' + Y_files[0] + '"does not exist')

    d0 = Y0.get_data()

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTIsize = sys.getsizeof(np.zeros(d0.shape,dtype='uint64'))

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use.
    blksize = int(np.floor(MAXMEM/8/NIFTIsize));

    # Reduce X to X for this block.
    X = pandas.io.parsers.read_csv(
        inputs['X'], sep=',', header=None).values
    X = X[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]

    # Mask volumes (if they are given)
    if 'M_files' in inputs:

        with open(inputs['M_files']) as a:

            M_files = []
            i = 0
            for line in a.readlines():

                M_files.append(line.replace('\n', ''))

        # If we have a mask for each Y, reduce the list to just for this block
        if len(M_files) == len(Y_files):

            # In this case we have a mask per Y volume
            M_files = M_files[(blksize*(batchNo-1)):min((blksize*batchNo),len(M_files))]
            MperY = True

        else:

            # Otherwise we have a mask/set of masks to apply to every Y.
            MperY = False

    # Otherwise we have no masks
    else:

        # There is not a mask for each Y as there are no masks at all!
        MperY = False
        M_files = None

    # Reduce Y_files to only Y files for this block
    Y_files = Y_files[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]
    
    # Verify input
    verifyInput(Y_files, M_files, Y0)

    # Obtain Y, mask for Y and nmap. This mask is just for voxels
    # with no studies present.
    Y, Mask, nmap = obtainY(Y_files, M_files, MperY)

    # Work out voxel specific designs
    MX = blkMX(X, Y)
    
    # Get X transpose Y, X transpose X and Y transpose Y.
    XtY = blkXtY(X, Y, Mask)
    YtY = blkYtY(Y, Mask)

    # In a spatially varying design XtX has dimensions n_voxels
    # by n_parameters by n_parameters. We reshape to n_voxels by
    # n_parameters^2 so that we can save as a csv.
    XtX_m = blkXtX(MX)
    XtX_m = XtX_m.reshape([XtX_m.shape[0], XtX_m.shape[1]*XtX_m.shape[2]])

    # We then need to unmask XtX as we now are saving XtX.
    XtX = np.zeros([Mask.shape[0],XtX_m.shape[1]])
    XtX[np.flatnonzero(Mask),:] = XtX_m[:]

    # Pandas reads and writes files much more quickly with nrows <<
    # number of columns
    XtY = XtY.transpose()

    if len(args)==1:
        # Record XtX and XtY
        np.savetxt(os.path.join(OutDir,"tmp","XtX" + str(batchNo) + ".csv"), 
                   XtX, delimiter=",")
        np.savetxt(os.path.join(OutDir,"tmp","XtY" + str(batchNo) + ".csv"), 
                   XtY, delimiter=",") 
        np.savetxt(os.path.join(OutDir,"tmp","YtY" + str(batchNo) + ".csv"), 
                   YtY, delimiter=",") 
        # Get map of number of scans at voxel.
        nmap = nib.Nifti1Image(nmap,
                               Y0.affine,
                               header=Y0.header)
        nib.save(nmap, os.path.join(OutDir,'tmp',
                        'blm_vox_n_batch'+ str(batchNo) + '.nii'))
    else:
        # Return XtX, XtY, YtY, nB
        return(XtX, XtY, YtY, nmap)
    w.resetwarnings()

def verifyInput(Y_files, M_files, Y0):

    # Obtain information about zero-th scan
    d0 = Y0.get_data()
    Y0aff = Y0.affine

    # Initial checks for NIFTI compatability for Y.
    for i in range(0, len(Y_files)):

        Y_file = Y_files[i]

        try:
            Y = nib.load(Y_file)
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
                M = nib.load(M_file)
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

def blkMX(X,Y):

    # Work out the mask.
    M = (Y!=0)

    # Get M in a form where each voxel's mask is mutliplied
    # by X
    M = M.transpose().reshape([M.shape[1], 1, M.shape[0]])
    Xt=X.transpose()

    # Obtain design for each voxel
    MXt = np.multiply(M, Xt)
    MX = MXt.transpose(0,2,1)

    return MX

def obtainY(Y_files, M_files, MperY):

    # Load in one nifti to check NIFTI size
    Y0 = nib.load(Y_files[0])
    d = Y0.get_data()
    
    # Get number of voxels.
    nvox = np.prod(d.shape)

    # Number of scans in block
    nscan = len(Y_files)

    # Count number of scans contributing to voxels
    nmap = np.zeros(d.shape)

    # Make an overall mask for the block if 
    # there's not a mask per individual
    if not MperY:

        if not M_files is None:

            M_overall = np.ones(d.shape)

            # Create intersect of all masks
            for M_file in M_files:

                M = nib.load(M_file).get_data()
                M_overall = np.multiply(M_overall, M)

    # Read in Y
    Y = np.zeros([nscan, nvox])
    for i in range(0, len(Y_files)):

        # Read in each individual NIFTI.
        Y_indiv = nib.load(Y_files[i])
        print(Y_indiv.shape)

        # Mask Y if necesart
        if M_files is not None:
            # If theres a mask for each individual load it
            if MperY:
                M_indiv = nib.load(M_files[i]).get_data()
            # Else apply group mask to each individual
            else:
                M_indiv = M_overall

            d = np.multiply(
                Y_indiv.get_data(),
                M_indiv)
        else: 
            #Just load in Y
            d = Y_indiv.get_data()

        # NaN check
        d = np.nan_to_num(d)

        # Count number of scans at each voxel
        nmap = nmap + 1*(np.nan_to_num(d)!=0)

        # Constructing Y matrix
        Y[i, :] = d.reshape([1, nvox])
    
    Mask = np.zeros([nvox])
    Mask[np.where(np.count_nonzero(Y, axis=0)>0)[0]] = 1
    
    Y = Y[:, np.where(np.count_nonzero(Y, axis=0)>0)[0]]

    return Y, Mask, nmap

# Note: this techniqcally calculates sum(Y.Y) for each voxel,
# not Y transpose Y for all voxels
def blkYtY(Y, Mask):

    # Read in number of scans and voxels.
    nscan = Y.shape[0]
    nvox = Y.shape[1]
    
    # Reshape Y
    Y_rs = Y.transpose().reshape(nvox, nscan, 1)
    Yt_rs = Y.transpose().reshape(nvox, 1, nscan)
    del Y

    # Calculate Y transpose Y.
    YtY_m = np.matmul(Yt_rs,Y_rs).reshape([nvox, 1])

    # Unmask YtY
    YtY = np.zeros([Mask.shape[0], 1])
    YtY[np.flatnonzero(Mask),:] = YtY_m[:]

    return YtY


def blkXtY(X, Y, Mask):
    
    # Calculate X transpose Y (Masked)
    XtY_m = np.asarray(
                np.dot(np.transpose(X), Y))

    # Check the dimensions haven't been reduced
    # (numpy will lower the dimension of the 
    # array if the length in one dimension is
    # one)
    if np.ndim(XtY_m) == 0:
        XtY_m = np.array([[XtY_m]])
    elif np.ndim(XtY_m) == 1:
        XtY_m = np.array([XtY_m])

    # Unmask XtY
    XtY = np.zeros([XtY_m.shape[0], Mask.shape[0]])
    XtY[:,np.flatnonzero(Mask)] = XtY_m[:]

    return XtY


def blkXtX(X):

    if np.ndim(X) == 3:

        Xt = X.transpose((0, 2, 1))
        XtX = np.matmul(Xt, X)

    else:

        # Calculate XtX
        XtX = np.asarray(
                    np.dot(np.transpose(X), X))

        # Check the dimensions haven't been reduced
        # (numpy will lower the dimension of the 
        # array if the length in one dimension is
        # one)
        if np.ndim(XtX) == 0:
            XtX = np.array([XtX])
        elif np.ndim(XtX) == 1:
            XtX = np.array([XtX])

    return XtX


if __name__ == "__main__":
    main()
