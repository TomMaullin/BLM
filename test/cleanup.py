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
from matplotlib import pyplot as plt


def cleanup(OutDir,simNo):

    # -----------------------------------------------------------------------
    # Get simulation directory
    # -----------------------------------------------------------------------
    # Simulation directory
    simDir = os.path.join(OutDir, 'sim' + str(simNo))

    # -----------------------------------------------------------------------
    # Create results directory (if we are on the first simulation)
    # -----------------------------------------------------------------------
    # Results directory
    resDir = os.path.join(OutDir,'results')

    # If resDir doesn't exist, make it
    if not os.path.exists(resDir):
        os.mkdir(resDir)

    # -----------------------------------------------------------------------
    # Remove data directory
    # -----------------------------------------------------------------------
    shutil.rmtree(os.path.join(simDir, 'data'))

# MARKER UP TO HERE

    # # -----------------------------------------------------------------------
    # # Remove BLM maps we are not interested in (for memory purposes)
    # # -----------------------------------------------------------------------
    # os.remove(os.path.join(simDir, 'BLM', 'blm_vox_con.nii'))
    # os.remove(os.path.join(simDir, 'BLM', 'blm_vox_conSE.nii'))
    # os.remove(os.path.join(simDir, 'BLM', 'blm_vox_conT.nii'))
    # os.remove(os.path.join(simDir, 'BLM', 'blm_vox_conT_swedf.nii'))
    # os.remove(os.path.join(simDir, 'BLM', 'blm_vox_edf.nii'))
    # os.remove(os.path.join(simDir, 'BLM', 'blm_vox_mask.nii'))

    # -----------------------------------------------------------------------
    # N map
    # -----------------------------------------------------------------------

    # Get spatially varying n
    n_sv = nib.load(os.path.join(simDir, 'BLM', 'blm_vox_n.nii')).get_data()

    # Work out number of subjects
    n = np.amax(n_sv)

    # Work out which voxels had readings for all subjects
    loc_sv = (n_sv>n//2)&(n_sv<n)
    loc_nsv = (n_sv==n)

    # Work out number of spatially varying and non-spatially varying voxels
    v_sv = np.sum(loc_sv)
    v_nsv = np.sum(loc_nsv)

    # Make line to add to csv for MRD
    n_line = np.array([[simNo, v_sv + v_nsv, v_sv, v_nsv]])

    # Number of voxels
    fname_n = os.path.join(resDir, 'n.csv')

    # Add to files 
    addLineToCSV(fname_n, n_line)

    # -----------------------------------------------------------------------
    # MAE and MRD for beta maps
    # -----------------------------------------------------------------------

    # Get BLM beta
    beta_blm = nib.load(os.path.join(simDir, 'BLM', 'blm_vox_beta.nii')).get_data()

    # Get lmer beta
    beta_lmer = nib.load(os.path.join(simDir, 'lmer', 'lmer_vox_beta.nii')).get_data()

    # Remove zero values (spatially varying)
    beta_blm_sv = beta_blm[(beta_lmer!=0) & loc_sv]
    beta_lmer_sv = beta_lmer[(beta_lmer!=0) & loc_sv]

    # Remove zero values (non spatially varying)
    beta_blm_nsv = beta_blm[(beta_lmer!=0) & loc_nsv]
    beta_lmer_nsv = beta_lmer[(beta_lmer!=0) & loc_nsv]

    # Remove zero values (both)
    beta_blm = beta_blm[beta_lmer!=0]
    beta_lmer = beta_lmer[beta_lmer!=0]

    # Get MAE
    MAE_beta = np.mean(np.abs(beta_blm-beta_lmer))
    MAE_beta_sv = np.mean(np.abs(beta_blm_sv-beta_lmer_sv))
    MAE_beta_nsv = np.mean(np.abs(beta_blm_nsv-beta_lmer_nsv))

    # Get MRD
    MRD_beta = np.mean(2*np.abs((beta_blm-beta_lmer)/(beta_blm+beta_lmer)))
    MRD_beta_sv = np.mean(2*np.abs((beta_blm_sv-beta_lmer_sv)/(beta_blm_sv+beta_lmer_sv)))
    MRD_beta_nsv = np.mean(2*np.abs((beta_blm_nsv-beta_lmer_nsv)/(beta_blm_nsv+beta_lmer_nsv)))

    # Make line to add to csv for MAE
    MAE_beta_line = np.array([[simNo, MAE_beta, MAE_beta_sv, MAE_beta_nsv]])

    # Make line to add to csv for MRD
    MRD_beta_line = np.array([[simNo, MRD_beta, MRD_beta_sv, MRD_beta_nsv]])

    # MAE beta file name
    fname_MAE = os.path.join(resDir, 'MAE_beta.csv')

    # MRD beta file name
    fname_MRD = os.path.join(resDir, 'MRD_beta.csv')

    # Add to files 
    addLineToCSV(fname_MAE, MAE_beta_line)
    addLineToCSV(fname_MRD, MRD_beta_line)

    # Cleanup
    del beta_lmer, beta_blm, MAE_beta, MRD_beta, MAE_beta_line, MRD_beta_line

    # -----------------------------------------------------------------------
    # Sigma2 maps
    # -----------------------------------------------------------------------

    # Get BLM sigma2
    sigma2_blm = nib.load(os.path.join(simDir, 'BLM', 'blm_vox_sigma2.nii')).get_data()

    # Get lmer sigma2
    sigma2_lmer = nib.load(os.path.join(simDir, 'lmer', 'lmer_vox_sigma2.nii')).get_data()

    # Remove zero values (spatially varying)
    sigma2_blm_sv = sigma2_blm[(sigma2_lmer!=0) & loc_sv]
    sigma2_lmer_sv = sigma2_lmer[(sigma2_lmer!=0) & loc_sv]

    # Remove zero values (non spatially varying)
    sigma2_blm_nsv = sigma2_blm[(sigma2_lmer!=0) & loc_nsv]
    sigma2_lmer_nsv = sigma2_lmer[(sigma2_lmer!=0) & loc_nsv]

    # Remove zero values
    sigma2_blm = sigma2_blm[sigma2_lmer!=0]
    sigma2_lmer = sigma2_lmer[sigma2_lmer!=0]

    # Get MAE
    MAE_sigma2 = np.mean(np.abs(sigma2_blm-sigma2_lmer))
    MAE_sigma2_sv = np.mean(np.abs(sigma2_blm_sv-sigma2_lmer_sv))
    MAE_sigma2_nsv = np.mean(np.abs(sigma2_blm_nsv-sigma2_lmer_nsv))

    # Get MRD
    MRD_sigma2 = np.mean(2*np.abs((sigma2_blm-sigma2_lmer)/(sigma2_blm+sigma2_lmer)))
    MRD_sigma2_sv = np.mean(2*np.abs((sigma2_blm_sv-sigma2_lmer_sv)/(sigma2_blm_sv+sigma2_lmer_sv)))
    MRD_sigma2_nsv = np.mean(2*np.abs((sigma2_blm_nsv-sigma2_lmer_nsv)/(sigma2_blm_nsv+sigma2_lmer_nsv)))

    # Make line to add to csv for MAE
    MAE_sigma2_line = np.array([[simNo, MAE_sigma2, MAE_sigma2_sv, MAE_sigma2_nsv]])

    # Make line to add to csv for MRD
    MRD_sigma2_line = np.array([[simNo, MRD_sigma2, MRD_sigma2_sv, MRD_sigma2_nsv]])

    # MAE sigma2 file name
    fname_MAE = os.path.join(resDir, 'MAE_sigma2.csv')

    # MRD sigma2 file name
    fname_MRD = os.path.join(resDir, 'MRD_sigma2.csv')

    # Add to files 
    addLineToCSV(fname_MAE, MAE_sigma2_line)
    addLineToCSV(fname_MRD, MRD_sigma2_line)

    # Cleanup
    del sigma2_lmer, sigma2_blm, MAE_sigma2, MRD_sigma2, MAE_sigma2_line, MRD_sigma2_line

    # -----------------------------------------------------------------------
    # Log-likelihood mean absolute difference
    # -----------------------------------------------------------------------

    # Get BLM llh
    llh_blm = nib.load(os.path.join(simDir, 'BLM', 'blm_vox_llh.nii')).get_data()

    # Get lmer llh
    llh_lmer = nib.load(os.path.join(simDir, 'lmer', 'lmer_vox_llh.nii')).get_data()

    # Remove zero values (spatially varying)
    llh_blm_sv = llh_blm[(llh_lmer!=0) & loc_sv]
    llh_lmer_sv = llh_lmer[(llh_lmer!=0) & loc_sv]

    # Remove zero values (non spatially varying)
    llh_blm_nsv = llh_blm[(llh_lmer!=0) & loc_nsv]
    llh_lmer_nsv = llh_lmer[(llh_lmer!=0) & loc_nsv]

    # Remove zero values
    llh_blm = llh_blm[llh_lmer!=0]
    llh_lmer = llh_lmer[llh_lmer!=0]

    # Get maximum absolute difference
    MAD_llh = np.mean(np.abs(llh_blm-llh_lmer))
    MAD_llh_sv = np.mean(np.abs(llh_blm_sv-llh_lmer_sv))
    MAD_llh_nsv = np.mean(np.abs(llh_blm_nsv-llh_lmer_nsv))

    # Print a string describing the results for the llh comparison
    # TO DO

    # Cleanup
    del llh_lmer, llh_blm, MAD_llh, MAD_llh_line

    # -----------------------------------------------------------------------
    # Cleanup finished!
    # -----------------------------------------------------------------------

    print('----------------------------------------------------------------')
    print('Simulation instance ' + str(simNo) + ' complete!')
    print('----------------------------------------------------------------')


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
    # Get number of random effects, levels and random factors in design
    # -----------------------------------------------------------------------
    # Random factor variables.
    rfxmats = inputs['Z']

    # Number of random effects
    r = len(rfxmats)

    # Number of random effects for each factor, q
    nraneffs = []

    # Number of levels for each factor, l
    nlevels = []

    for k in range(r):

        rfxdes = loadFile(rfxmats[k]['f' + str(k+1)]['design'])
        rfxfac = loadFile(rfxmats[k]['f' + str(k+1)]['factor'])

        nraneffs = nraneffs + [rfxdes.shape[1]]
        nlevels = nlevels + [len(np.unique(rfxfac))]

    # Get number of random effects
    nraneffs = np.array(nraneffs)
    nlevels = np.array(nlevels)
    q = np.sum(nraneffs*nlevels)

    # Number of covariance parameters
    ncov = np.sum(nraneffs*(nraneffs+1)//2)

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
    beta_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lmer', 'beta_' + str(cv) + '.csv')).values

    print('beta_current shape', beta_current.shape)

    # Loop through parameters adding them one voxel at a time
    for param in np.arange(p):

        # Add back to a NIFTI file
        addBlockToNifti(os.path.join(simDir,"lmer","lmer_vox_beta.nii"), beta_current[:,param], inds_cv, volInd=param,dim=(*dim,int(p)))

    # Remove file
    os.remove(os.path.join(simDir, 'lmer', 'beta_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Sigma2 combine
    # -------------------------------------------------------------------

    # Read in file
    sigma2_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lmer', 'sigma2_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lmer","lmer_vox_sigma2.nii"), sigma2_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lmer', 'sigma2_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # vechD combine
    # -------------------------------------------------------------------

    # Read in file
    vechD_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lmer', 'vechD_' + str(cv) + '.csv')).values

    # Loop through covariance parameters adding them one voxel at a time
    for param in np.arange(ncov):

        # Add back to a NIFTI file
        addBlockToNifti(os.path.join(simDir,"lmer","lmer_vox_D.nii"), vechD_current[:,param], inds_cv, volInd=param,dim=(*dim,int(ncov)))

    # Remove file
    os.remove(os.path.join(simDir, 'lmer', 'vechD_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Log-likelihood combine
    # -------------------------------------------------------------------

    # Read in file
    llh_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lmer', 'llh_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lmer","lmer_vox_llh.nii"), llh_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lmer', 'llh_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # T statistic combine
    # -------------------------------------------------------------------

    # Read in file
    Tstat_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lmer', 'Tstat_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lmer","lmer_vox_conT.nii"), Tstat_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lmer', 'Tstat_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # P value combine
    # -------------------------------------------------------------------

    # Read in file
    Pval_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lmer', 'Pval_' + str(cv) + '.csv')).values

    # Change to log scale
    Pval_current[Pval_current!=0]=-np.log10(Pval_current[Pval_current!=0])

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lmer","lmer_vox_conTlp.nii"), Pval_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lmer', 'Pval_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Times combine
    # -------------------------------------------------------------------

    # Read in file
    times_current = pd.io.parsers.read_csv(os.path.join(simDir, 'lmer', 'times_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(simDir,"lmer","lmer_vox_times.nii"), times_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(simDir, 'lmer', 'times_' + str(cv) + '.csv'))

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