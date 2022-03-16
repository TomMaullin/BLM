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

def combine_batch_masking(*args):

    print('marker')

    t1 = time.time()
    print('started, time ',t1-t1)

    # Work out number of batchs
    n_b = args[0]

    # Current node
    node = args[1]

    # Number of nodes
    numNodes = args[2]

    maskJob = args[3]

    print('n_b: ', n_b, ', node: ', node, ', numNodes: ', numNodes, ', maskJob: ', maskJob)

    # ----------------------------------------------------------------------
    # Check inputs
    # ----------------------------------------------------------------------
    if len(args)==4 or (not args[4]):
        # Load in inputs
        with open(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..',
                    'blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream,Loader=yaml.FullLoader)
    else:
        if type(args[4]) is str:
            # In this case inputs file is first argument
            with open(os.path.join(args[4]), 'r') as stream:
                inputs = yaml.load(stream,Loader=yaml.FullLoader)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[4]

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

    # # Work out number of batchs
    # n_b = len(glob.glob(os.path.join(OutDir,"tmp","blm_vox_n_batch*")))

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
    
        # Empty loop
        emptyLoop = False

    elif ((1+(node-1)*n_images) <= (n_b + 1)):
    
        # Work out loop range
        loopRange = range(1+(node-1)*n_images,1+node*n_images)
    
        # This is not the last node
        lastNode = False

        # Empty loop
        emptyLoop = False

    else:

        # Empty loop range
        loopRange = range(0,0)

        # This is not the last node (this one's redundant)
        lastNode = False

        # Empty loop
        emptyLoop = True

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
        os.remove(os.path.join(OutDir,"tmp", "blm_vox_n_batch" + str(batchNo) + ".nii"))

    # Filename for nmap
    n_fname = os.path.join(OutDir,'blm_vox_n.nii')

    # Filename for dfmap
    df_fname = os.path.join(OutDir,'blm_vox_edf.nii')

    if not emptyLoop:

        # Check if file is in use
        fileLocked = True
        while fileLocked:
            try:
                # Create lock file, so other jobs know we are writing to this file
                f=os.open(os.path.join(OutDir,"config_write.lock"), os.O_CREAT|os.O_EXCL|os.O_RDWR)
                fileLocked = False
            except FileExistsError:
                fileLocked = True

        # ------------------------------------------------------------------------------------
        # MARKER ADD TO RUNNING TOTAL
        # ------------------------------------------------------------------------------------

        if os.path.exists(df_fname):
            df_sv = n_sv + loadFile(df_fname).get_fdata()
            os.remove(df_fname)
        else:
            df_sv = np.array(n_sv) # MARKER SOMETHING WRONG WITH DF

        if os.path.exists(n_fname):
            n_sv = n_sv + loadFile(n_fname).get_fdata()
            os.remove(n_fname)

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
        os.remove(os.path.join(OutDir,"config_write.lock"))
        os.close(f)
        
    # --------------------------------------------------------------------------------
    # Create Mask
    # --------------------------------------------------------------------------------

    if maskJob:

        # Read in degrees of freedom
        df_sv = loadFile(os.path.join(OutDir,'blm_vox_edf.nii')).get_fdata()

        # Remove non-zero voxels
        df_sv = np.maximum(df_sv,0)

        # Write to file
        dfmap = nib.Nifti1Image(df_sv,
                                nifti.affine,
                                header=nifti.header) 
        nib.save(dfmap, df_fname)
        del dfmap

        # Read in n (spatially varying)
        n_sv  = loadFile(os.path.join(OutDir,'blm_vox_n.nii')).get_fdata()

        Mask = np.ones([v, 1])
        n_sv = n_sv.reshape(v, 1)   

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


# ============================================================================
#
# 
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `AtBstr`: A string representing which product matrix we are looking at. 
#             e.g. "XtX" for X'X.
# - `OutDir`: Output directory.
# - 
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `AtB`: The product matrix (flattened); If we had wanted X'X (which is 
#          dimension p by p) for v voxels, the output would here would have 
#          dimension (1 by p**2). If sv was True, we will have one matrix for
#          each voxel. If sv was false we will have one matrix for all voxels.
#
# ============================================================================
def combine_batch_designs(AtBstr, OutDir, fileRange):

    # MARKER HANDLE EMPTY JOB

    # Get NIFTIfilenames
    NIFTIfilenames = [os.path.join(OutDir,"tmp", 
        "blm_vox_uniqueM_batch" + str(i) + ".nii") for i in fileRange]

    # Get AtB filenames
    AtBfilenames = [os.path.join(OutDir,"tmp",
        AtBstr+ str(i) + ".npy") for i in fileRange]

    # Affine and header
    aff = loadFile(NIFTIfilenames[0]).affine
    hdr = loadFile(NIFTIfilenames[0]).header

    # Initialize current uniqueness map and AtB
    uniquenessMask_current = loadFile(NIFTIfilenames[0]).get_fdata()
    AtB_unique_current = np.load(AtBfilenames[0])

    # Add row of zeros for outside of mask
    AtB_unique_current = np.concatenate((np.zeros((1,AtB_unique_current.shape[1])), AtB_unique_current), axis=0)

    # Remove files
    os.remove(NIFTIfilenames[0])
    os.remove(AtBfilenames[0])

    # Loop through other files reading one at a time.
    for i in np.arange(1, len(NIFTIfilenames)+1):

        # Loop through assigned images
        if i < len(NIFTIfilenames):

            # Read new uniqueness mask
            uniquenessMask_new = loadFile(NIFTIfilenames[i]).get_fdata()

            # Read in new unique list of AtB's
            AtB_unique_new = np.load(AtBfilenames[i])
        
            # Add row of zeros for outside of mask
            AtB_unique_new = np.concatenate((np.zeros((1,AtB_unique_new.shape[1])), AtB_unique_new), axis=0)

            # Remove files
            os.remove(NIFTIfilenames[i])
            os.remove(AtBfilenames[i])

        # Once we've done the assigned images we do the last image
        else:

            # Check if running totals in use
            fileLocked = True
            while fileLocked:
                try:
                    # Create lock file, so other jobs know we are writing to this file
                    f=os.open(os.path.join(OutDir,'tmp',AtBstr + ".lock"), os.O_CREAT|os.O_EXCL|os.O_RDWR)
                    fileLocked = False
                except FileExistsError:
                    fileLocked = True

            # Check whether the NIFTI exists already
            if os.path.isfile(os.path.join(OutDir,"tmp", "blm_vox_uniqueM.nii")):

                # Adding to the running total now
                uniquenessMask_new = loadFile(os.path.join(OutDir,"tmp", 
                                              "blm_vox_uniqueM.nii")).get_fdata()

                # Read in the running list of AtB's
                AtB_unique_new = np.load(os.path.join(OutDir,"tmp",
                                         AtBstr + ".npy"))

            # If this is the first time we've made the running total we don't need
            # to add it so we break out of the loop
            else:

                break

        # Get maxM for new uniqueness mask.
        maxM_new = np.int64(np.amax(uniquenessMask_new))

        # Get maxM for running uniqueness mask.
        maxM_current = np.int64(np.amax(uniquenessMask_current))

        # Get max of both
        maxM = np.maximum(maxM_new,maxM_current)

        # Get updated uniqueness mask
        uniquenessMask_updated_full = uniquenessMask_current*((maxM+1)**0) + uniquenessMask_new*((maxM+1)**1)

        # Get unique values in new map.
        uniqueVals = np.unique(uniquenessMask_updated_full, return_inverse = True)
        uniquenessMask_updated = uniqueVals[1].reshape(uniquenessMask_new.shape)
        uniqueVal_array = uniqueVals[0]

        # Update maxM
        maxM_updated = np.amax(uniquenessMask_updated)

        # Initialise AtB_unique_updated
        AtB_unique_updated = np.zeros([maxM_updated+1, AtB_unique_current.shape[1]])

        # Get value1 and value2, corresponding to values in original uniqueness maps.
        for value_updated in np.arange(maxM_updated+1):

            # Get value we're interested in
            value_updated_full = uniqueVal_array[value_updated]

            # Work out which value the updated uniqueness values corresponded to in the `new' image
            value_new = int(value_updated_full//(maxM+1))

            # Work out which value the updated uniqueness values corresponded to in the `current' image
            value_current = int(value_updated_full-value_new*(maxM+1))

            # Update the unique AtB array
            AtB_unique_updated[value_updated,:] = AtB_unique_new[value_new,:] + AtB_unique_current[value_current,:]

        # Update uniquenessMask_current as uniquenessMask_updated
        uniquenessMask_current = np.array(uniquenessMask_updated)

        # Update AtB_unique_current as AtB_unique_updated
        AtB_unique_current = np.array(AtB_unique_updated)

    # Make nifti
    uniquenessMask_current = nib.Nifti1Image(uniquenessMask_current, 
                                             aff, header=hdr)

    # Save uniqueness map
    nib.save(uniquenessMask_current, os.path.join(OutDir,"tmp", 
             "blm_vox_uniqueM.nii"))

    # Save unique designs
    np.save(os.path.join(OutDir,"tmp",AtBstr),AtB_unique_current)

    # Delete lock file, so other jobs know they can now write to the
    # file
    os.remove(os.path.join(OutDir,'tmp',AtBstr + ".lock"))
    os.close(f)

    
if __name__ == "__rain__":
    main()