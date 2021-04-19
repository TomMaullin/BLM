import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import scipy
import scipy.sparse
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
from scipy import ndimage
import time
import pandas as pd
from lib.fileio import *
from matplotlib import pyplot as plt
from statsmodels.stats import multitest


# ===========================================================================
#
# Inputs:
#
# ---------------------------------------------------------------------------
#
# ===========================================================================
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

    # -----------------------------------------------------------------------
    # Remove BLM maps we are not interested in (for memory purposes)
    # -----------------------------------------------------------------------
    os.remove(os.path.join(simDir, 'BLM', 'blm_vox_con.nii'))
    os.remove(os.path.join(simDir, 'BLM', 'blm_vox_conSE.nii'))
    os.remove(os.path.join(simDir, 'BLM', 'blm_vox_conT.nii'))
    os.remove(os.path.join(simDir, 'BLM', 'blm_vox_conT_swedf.nii'))
    os.remove(os.path.join(simDir, 'BLM', 'blm_vox_edf.nii'))
    os.remove(os.path.join(simDir, 'BLM', 'blm_vox_mask.nii'))

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
    # P values 
    # -----------------------------------------------------------------------
    # Load logp map
    logp = nib.load(os.path.join(simDir, 'BLM', 'blm_vox_conTlp.nii')).get_data()

    # Remove zeros
    logp = logp[logp!=0]

    # Un-"log"
    p = 10**(-logp)

    # Get bin counts
    counts,_,_=plt.hist(p, bins=100, label='hist')

    # Make line to add to csv for bin counts
    pval_line = np.concatenate((np.array([[simNo]]),np.array([counts])),axis=1)

    # pval file name
    fname_pval = os.path.join(resDir, 'pval_counts.csv')

    # Add to files 
    addLineToCSV(fname_pval, pval_line)

    # Convert to one tailed
    p_ot = np.zeros(p.shape)
    p_ot[p<0.5] = 2*p[p<0.5]
    p_ot[p>0.5] = 2*(1-p[p>0.5])
    p = p_ot

    # Perform bonferroni
    fwep_bonferroni = multitest.multipletests(p,alpha=0.05,method='bonferroni')[0]

    # Get number of false positives
    fwep_bonferroni = np.sum(fwep_bonferroni)

    # Make line to add to csv for fwe
    fwe_line = np.concatenate((np.array([[simNo]]),
                               np.array([[fwep_bonferroni]])),axis=1)

    # pval file name
    fname_fwe = os.path.join(resDir, 'pval_fwe.csv')

    # Add to files 
    addLineToCSV(fname_fwe, fwe_line)

    # Cleanup
    del p, logp, counts, fname_pval, pval_line, fname_fwe, fwe_line

    # -----------------------------------------------------------------------
    # Cleanup finished!
    # -----------------------------------------------------------------------

    print('----------------------------------------------------------------')
    print('Simulation instance ' + str(simNo) + ' complete!')
    print('----------------------------------------------------------------')


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
