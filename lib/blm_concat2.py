import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
import time
import warnings
import subprocess
np.set_printoptions(threshold=sys.maxsize)
from scipy import stats
from lib.fileio import *

# Developer notes:
# --------------------------------------------------------------------------
# In the following code I have used the following subscripts to indicate:
#
# _r - This means this is an array of values corresponding to voxels which
#      are present in between k and n-1 studies (inclusive), where k is
#      decided by the user specified thresholds. These voxels will typically
#      be on the edge of the brain and look like a "ring" around the brain,
#      hence "_r" for ring.
# 
# _i - This means that this is an array of values corresponding to voxels 
#      which are present in all n studies. These will usually look like
#      a smaller mask place inside the whole study mask. Hence "_i" for 
#      inner.
#
# _sv - This means this variable is spatially varying (There is a reading
#       per voxel). 
#
# --------------------------------------------------------------------------
# Author: Tom Maullin (04/02/2019)

def main3(*args):

    print('marker')

    t1 = time.time()
    print('started, time ',t1-t1)

    # Work out number of batchs
    n_b = args[0]

    # Current node
    node = args[1]

    # Number of nodes
    numNodes = args[2]

    # ----------------------------------------------------------------------
    # Check inputs
    # ----------------------------------------------------------------------
    if len(args)==3 or (not args[3]):
        # Load in inputs
        with open(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..',
                    'blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream,Loader=yaml.FullLoader)
    else:
        if type(args[3]) is str:
            # In this case inputs file is first argument
            with open(os.path.join(args[3]), 'r') as stream:
                inputs = yaml.load(stream,Loader=yaml.FullLoader)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[3]

    t2 = time.time()
    print('inputs, time ',t2-t1)

    # ----------------------------------------------------------------------
    # Read basic inputs
    # ----------------------------------------------------------------------
    OutDir = inputs['outdir']

    # Get number of fixed effects parameters
    L1 = str2vec(inputs['contrasts'][0]['c' + str(1)]['vector'])
    L1 = np.array(L1)
    p = L1.shape[0]
    del L1
    
    # Read in the nifti size and work out number of voxels.
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = loadFile(nifti_path)

    NIFTIsize = nifti.shape
    v = int(np.prod(NIFTIsize))

    # Check if the maximum memory is saved.    
    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    # Work out if we are outputting cov(beta) maps
    if "OutputCovB" in inputs:
        OutputCovB = inputs["OutputCovB"]
    else:
        OutputCovB = True

    t2 = time.time()
    print('inputs2, time ',t2-t1)

    print('OutputCovB ', OutputCovB)

    # --------------------------------------------------------------------------------
    # Get n (number of observations) and n_sv (spatially varying number of
    # observations)
    # --------------------------------------------------------------------------------

    print('marker2')

    # Work out number of batchs
    n_b = len(glob.glob(os.path.join(OutDir,"tmp","blm_vox_n_batch*")))

    # ----------------------------------------------------------------
    # CHANGED
    # ----------------------------------------------------------------

    # Number of images to look at for each node 
    n_images = n_b//numNodes+1

    # ----------------------------------------------------------------
    # Remove file we just read
    #os.remove(os.path.join(OutDir,"tmp", "blm_vox_n_batch1.nii"))

    print('marker3')

    # MARKER: node range is 1 to numNodes


    # REMEM TO START LOOP WITH 1 MAP ABOVE
    # Becomes: 

    # Loop over:
    # 

    if ((1 + node*n_images) >= (n_b + 1)) and ((1+(node-1)*n_images) <= (n_b + 1)):
    
        # Work out loop range
        loopRange = range(1+(node-1)*n_images,(n_b+1))
    
        # This is the last node
        lastNode = True
        redundant = False
    
    elif ((1+(node-1)*n_images) <= (n_b + 1)):
    
        # Work out loop range
        loopRange = range(1+(node-1)*n_images,1+node*n_images)
    
        # This is not the last node
        lastNode = False
        redundant = False

    else:

        # Empty loop range
        loopRange = range(0,0)

        # This is not the last node (this one's redundant)
        lastNode = False
        redundant = True

    if not redundant:

        # Check if this is the first image we're looking at
        firstImage = True

        # Cycle through batches and add together n.
        for batchNo in loopRange:

            print('batchNo: ', batchNo)

            if firstImage:

                # Read in n (spatially varying)
                n_sv  = loadFile(os.path.join(OutDir,"tmp", 
                                 "blm_vox_n_batch" + str(batchNo) + ".nii")).get_fdata()

                # No longer looking at the first image
                firstImage = False

            else:

                # Obtain the full nmap.
                n_sv = n_sv + loadFile(os.path.join(OutDir,"tmp", 
                    "blm_vox_n_batch" + str(batchNo) + ".nii")).get_fdata()

            # Remove file we just read
            # os.remove(os.path.join(OutDir,"tmp", "blm_vox_n_batch" + str(batchNo) + ".nii"))

        # Filename for nmap
        n_fname = os.path.join(OutDir,'blm_vox_n.nii')

        # Filename for dfmap
        df_fname = os.path.join(OutDir,'blm_vox_edf.nii')

        # Check if file is in use
        fileLocked = True
        while fileLocked:
            try:
                # Create lock file, so other jobs know we are writing to this file
                f=os.open(n_fname + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
                fileLocked = False
            except FileExistsError:
                fileLocked = True

        
        # ------------------------------------------------------------------------------------
        # MARKER ADD TO RUNNING TOTAL
        # ------------------------------------------------------------------------------------
        if os.path.exists(n_fname):
            n_sv = n_sv + loadFile(n_fname).get_fdata()
            os.remove(n_fname)

        if os.path.exists(df_fname):
            df_sv = n_sv + loadFile(df_fname).get_fdata()
            os.remove(df_fname)

        # Save nmap
        nmap = nib.Nifti1Image(n_sv,
                               nifti.affine,
                               header=nifti.header)
        nib.save(nmap, n_fname)
        del nmap

        # Save dfmap
        if not lastNode:
            dfmap = nib.Nifti1Image(df_sv,
                                    nifti.affine,
                                    header=nifti.header)
        else:
            dfmap = nib.Nifti1Image(df_sv-p,
                                    nifti.affine,
                                    header=nifti.header)

        nib.save(dfmap, df_fname)
        del dfmap

        # Delete lock file, so other jobs know they can now write to the
        # file
        os.remove(n_fname + ".lock")
        os.close(f)
        
        # --------------------------------------------------------------------------------
        # Create Mask
        # --------------------------------------------------------------------------------

        if lastNode:

            Mask = np.ones([v, 1])

            # Check for user specified missingness thresholds.
            if 'Missingness' in inputs:

                # Apply user specified missingness thresholding.
                if ("MinPercent" in inputs["Missingness"]) or ("minpercent" in inputs["Missingness"]):

                    # Read in relative threshold
                    if "MinPercent" in inputs["Missingness"]:
                        rmThresh = inputs["Missingness"]["MinPercent"]
                    else:
                        rmThresh = inputs["Missingness"]["minpercent"]

                    # If it's a percentage it will be a string and must be converted.
                    rmThresh = str(rmThresh)
                    if "%" in rmThresh:
                        rmThresh = float(rmThresh.replace("%", ""))/100
                    else:
                        rmThresh = float(rmThresh)

                    # Check the Relative threshold is between 0 and 1.
                    if (rmThresh < 0) or (rmThresh > 1):
                        raise ValueError('Minumum percentage missingness threshold is out of range: ' +
                                         '0 < ' + str(rmThresh) + ' < 1 violation')

                    # Mask based on threshold.
                    Mask[n_sv<rmThresh*n]=0

                if ("MinN" in inputs["Missingness"]) or ("minn" in inputs["Missingness"]):

                    # Read in relative threshold
                    if "minn" in inputs["Missingness"]:
                        amThresh = inputs["Missingness"]["minn"]
                    else:
                        amThresh = inputs["Missingness"]["MinN"]

                    # If it's a percentage it will be a string and must be converted.
                    if isinstance(amThresh, str):
                        amThresh = float(amThresh)

                    # Mask based on threshold.
                    Mask[n_sv<amThresh]=0

            # We remove anything with 1 degree of freedom (or less) by default.
            # 1 degree of freedom seems to cause broadcasting errors on a very
            # small percentage of voxels.
            Mask[n_sv<=p+1]=0

            if 'analysis_mask' in inputs:

                amask_path = inputs["analysis_mask"]
                
                # Read in the mask nifti.
                amask = loadFile(amask_path).get_fdata().reshape([v,1])

            else:

                # By default make amask ones
                amask = np.ones([v,1])


            # Get indices for whole analysis mask. These indices are the indices we
            # have recorded for the product matrices with respect to the entire volume
            amInds = get_amInds(amask)
                
            # Ensure overall mask matches analysis mask
            Mask[~np.in1d(np.arange(v).reshape(v,1), amInds)]=0

            # Output final mask map
            maskmap = nib.Nifti1Image(Mask.reshape(
                                            NIFTIsize[0],
                                            NIFTIsize[1],
                                            NIFTIsize[2]
                                            ),
                                      nifti.affine,
                                      header=nifti.header) 
            nib.save(maskmap, os.path.join(OutDir,'blm_vox_mask.nii'))
            del maskmap

        # t2 = time.time()
        # print('mask, time ',t2-t1)


        # # Unmask df
        # df = np.zeros([v])
        # df[R_inds] = df_r 
        # df[I_inds] = df_i

        # df = df.reshape(int(NIFTIsize[0]),
        #                 int(NIFTIsize[1]),
        #                 int(NIFTIsize[2]))

        # # Save beta map.
        # dfmap = nib.Nifti1Image(df,
        #                         nifti.affine,
        #                         header=nifti.header) 
        # nib.save(dfmap, os.path.join(OutDir,'blm_vox_edf.nii'))
        # del df, dfmap

        # t2 = time.time()
        # print('vedf, time ',t2-t1)


    
if __name__ == "__rain__":
    main()
