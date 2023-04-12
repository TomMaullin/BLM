import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import nibabel as nib
import os
import shutil
import yaml
import pandas as pd
from lib.fileio import *

def cleanup(out_dir,sim_ind):

    # -----------------------------------------------------------------------
    # Get simulation directory
    # -----------------------------------------------------------------------
    # Simulation directory
    sim_dir = os.path.join(out_dir, 'sim' + str(sim_ind))

    # -----------------------------------------------------------------------
    # Create results directory (if we are on the first simulation)
    # -----------------------------------------------------------------------
    # Results directory
    res_dir = os.path.join(sim_dir,'results')

    # If resDir doesn't exist, make it
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
        
    # -----------------------------------------------------------------------
    # Remove data directory
    # -----------------------------------------------------------------------
    if os.path.exists(os.path.join(sim_dir, 'data')):
        shutil.rmtree(os.path.join(sim_dir, 'data'))
    
    # -----------------------------------------------------------------------
    # N map
    # -----------------------------------------------------------------------

    # Get spatially varying n
    n_sv = nib.load(os.path.join(sim_dir, 'BLM', 'blm_vox_n.nii')).get_fdata()

    # Work out number of subjects
    n = np.amax(n_sv)

    # Work out which voxels had readings for all subjects
    loc_sv = (n_sv>n//2)&(n_sv<n)
    loc_nsv = (n_sv==n)

    # Work out number of spatially varying and non-spatially varying voxels
    v_sv = np.sum(loc_sv)
    v_nsv = np.sum(loc_nsv)
    
    # -----------------------------------------------------------------------
    # Files to compare
    # -----------------------------------------------------------------------
    
    filenames = ['blm_vox_beta.nii', 'blm_vox_conT.nii', 'blm_vox_conTlp.nii',
                 'blm_vox_llh.nii', 'blm_vox_resms.nii']
    # -----------------------------------------------------------------------
    # Compare to lm
    # -----------------------------------------------------------------------
    
    # Loop through files
    for blm_filename in filenames:
        
        # Get the equivalent lm file
        lm_filename = blm_filename.replace('blm', 'lm')
        
        # Make the filenames full
        blm_filename = os.path.join(sim_dir, 'BLM', blm_filename)
        lm_filename = os.path.join(sim_dir, 'lm', lm_filename)
        
        # We need to create the lm residual mean squares map using sigma2 and n
        if blm_filename == os.path.join(sim_dir, 'BLM', 'blm_vox_resms.nii'):
                 
            # Load in lm sigma2 map
            lm_sigma2 = nib.load(os.path.join(sim_dir, 'lm', 'lm_vox_sigma2.nii')).get_fdata()

            # Multiply by the spatially varying n map
            lm_resms = lm_sigma2 * n_sv
            
            print(np.sum(np.isnan(n_sv)))
            print(np.sum(np.isnan(lm_sigma2)))
            print(np.sum(np.isnan(lm_resms)))

            # Output as the lm resms map
            nib.save(nib.Nifti1Image(lm_resms, np.eye(4)), lm_filename)
        
        # -------------------------------------------------------------------
        # Read in files
        # -------------------------------------------------------------------

        # Get BLM map
        blm = nib.load(blm_filename).get_fdata()
        print(np.sum(np.isnan(blm)))

        # Get lm map
        lm = nib.load(lm_filename).get_fdata()
        print(np.sum(np.isnan(lm)))

        # Remove zero values (spatially varying)
        blm_sv = blm[(lm!=0) & loc_sv]
        lm_sv = lm[(lm!=0) & loc_sv]

        # Remove zero values (non spatially varying)
        blm_nsv = blm[(lm!=0) & loc_nsv]
        lm_nsv = lm[(lm!=0) & loc_nsv]

        # Remove zero values (both)
        blm = blm[lm!=0]
        lm = lm[lm!=0]

        # Get MAE
        MAE = np.mean(np.abs(blm-lm))
        MAE_sv = np.mean(np.abs(blm_sv-lm_sv))
        MAE_nsv = np.mean(np.abs(blm_nsv-lm_nsv))

        # Test results
        result_MAE = 'Pass' if MAE < 1e-6 else 'Fail'
        result_MAE_sv = 'Pass' if MAE_sv < 1e-6 else 'Fail'
        result_MAE_nsv = 'Pass' if MAE_nsv < 1e-6 else 'Fail'

        # Get MRD
        MRD = np.mean(2*np.abs((blm-lm)/(blm+lm)))
        MRD_sv = np.mean(2*np.abs((blm_sv-lm_sv)/(blm_sv+lm_sv)))
        MRD_nsv = np.mean(2*np.abs((blm_nsv-lm_nsv)/(blm_nsv+lm_nsv)))

        # Test results
        result_MRD = 'Pass' if MRD < 1e-6 else 'Fail'
        result_MRD_sv = 'Pass' if MRD_sv < 1e-6 else 'Fail'
        result_MRD_nsv = 'Pass' if MRD_nsv < 1e-6 else 'Fail'

        # Print results
        print('----------------------------------------------------------------------------------------------')
        print('Test Results for ' + str(blm_filename))
        print('----------------------------------------------------------------------------------------------')
        print(' ')
        print('Mean Absolute Errors: ')
        print('    All voxels: ' + repr(MAE) + ', Result: ' + result_MAE)
        print('    Spatially varying voxels: ' + repr(MAE_sv) + ', Result: ' + result_MAE_sv)
        print('    Non-spatially varying voxels: ' + repr(MAE_nsv) + ', Result: ' + result_MAE_nsv)
        print(' ')
        print('Mean Relative Differences: ')
        print('    All voxels: ' + repr(MRD) + ', Result: ' + result_MRD)
        print('    Spatially varying voxels: ' + repr(MRD_sv) + ', Result: ' + result_MRD_sv)
        print('    Non-spatially varying voxels: ' + repr(MRD_nsv) + ', Result: ' + result_MRD_nsv)
        print(' ')
        
    print('----------------------------------------------------------------------------------------------')


# Add R output to nifti files
def Rcleanup(OutDir, simNo, nvg, cv):

    # Get simulation directory
    simDir = os.path.join(OutDir, 'sim' + str(simNo))

    # -----------------------------------------------------------------------
    # Read in design in BLM inputs form (this just is easier as code already
    # exists for using this format).
    # -----------------------------------------------------------------------
    # There should be an inputs file in each simulation directory
    with open(os.path.join(simDir,'inputs.yml'), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # -----------------------------------------------------------------------
    # Get number of observations and fixed effects
    # -----------------------------------------------------------------------
    X = pd.io.parsers.read_csv(os.path.join(simDir,"data","X.csv"), header=None).values
    n = X.shape[0]
    p = X.shape[1]

    # -----------------------------------------------------------------------
    # Get number voxels and dimensions
    # -----------------------------------------------------------------------

    # nmap location 
    nmap = os.path.join(simDir, "data", "Y0.nii")

    # Work out dim if we don't already have it
    dim = nib.Nifti1Image.from_filename(nmap, mmap=False).shape[:3]

    # Work out affine
    affine = nib.Nifti1Image.from_filename(nmap, mmap=False).affine.copy()

    # Number of voxels
    v = np.prod(dim)

    # Delete nmap
    del nmap
    
    # -------------------------------------------------------------------
    # Voxels of interest
    # -------------------------------------------------------------------

    # Split voxels we want to look at into groups we can compute
    voxelGroups = np.array_split(np.arange(v), nvg)

    # Current group of voxels
    inds_cv = voxelGroups[cv]

    # Number of voxels currently
    v_current = len(inds_cv)

    # -------------------------------------------------------------------
    # Beta combine
    # -------------------------------------------------------------------

    # Read in file
    beta_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lm', 'beta_' + str(cv) + '.csv')).values

    # Loop through parameters adding them one voxel at a time
    for param in np.arange(p):

        # Add back to a NIFTI file
        addBlockToNifti(os.path.join(simDir,"lm","lm_vox_beta.nii"), beta_current[:,param], inds_cv, volInd=param,dim=(*dim,int(p)))

    # Remove file
    os.remove(os.path.join(simDir, 'lm', 'beta_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Sigma2 combine
    # -------------------------------------------------------------------

    # Read in file
    sigma2_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lm', 'sigma2_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lm","lm_vox_sigma2.nii"), sigma2_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lm', 'sigma2_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Log-likelihood combine
    # -------------------------------------------------------------------

    # Read in file
    llh_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lm', 'llh_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lm","lm_vox_llh.nii"), llh_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lm', 'llh_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # T statistic combine
    # -------------------------------------------------------------------

    # Read in file
    Tstat_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lm', 'Tstat_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lm","lm_vox_conT.nii"), Tstat_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lm', 'Tstat_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # P value combine
    # -------------------------------------------------------------------

    # Read in file
    Pval_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lm', 'Pval_' + str(cv) + '.csv')).values

    # Change to log scale
    Pval_current[Pval_current!=0]=-np.log10(Pval_current[Pval_current!=0])

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lm","lm_vox_conTlp.nii"), Pval_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lm', 'Pval_' + str(cv) + '.csv'))


# This function adds a line to a csv. If the csv does not exist it creates it.
# It uses a filelock system
def addLineToCSV(fname, line):

    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            f = os.open(fname + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True

    # Check if file already exists and if so read it in
    if os.path.isfile(fname):

        # Read in data
        data = pd.io.parsers.read_csv(fname, header=None, index_col=None).values

        # Append line to data
        data = np.concatenate((data, line),axis=0)

    else:

        # The data is just this line
        data = line

    # Write data back to file
    pd.DataFrame(data).to_csv(fname, header=None, index=None)

    # Delete lock file, so other jobs know they can now write to the
    # file
    os.remove(fname + ".lock")
    os.close(f)

    del fname