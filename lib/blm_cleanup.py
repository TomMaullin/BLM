import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
import time
np.set_printoptions(threshold=sys.maxsize)

# ====================================================================================
#
# This file is the cleanup stage of the BLM pipeline. It simply deletes any remaining
# files that are no longer needed.
#
# ------------------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited: 04/04/2020)
#
# ------------------------------------------------------------------------------------
#
# The code takes the following inputs:
#
#  - `ipath`: Path to an `inputs` yml file, following the same formatting guidelines
#             as `blm_config.yml`. 
#
# ====================================================================================
def main4(ipath):

    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Inputs file is first argument
    with open(os.path.join(ipath), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # --------------------------------------------------------------------------------
    # Read Output directory, work out number of batches
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']

    # --------------------------------------------------------------------------------
    # Clean up files
    # --------------------------------------------------------------------------------
    os.remove(os.path.join(OutDir, 'nb.txt'))
    if os.path.isdir(os.path.join(OutDir, 'tmp')):
        shutil.rmtree(os.path.join(OutDir, 'tmp'))


    print('Analysis complete!')
    print('')
    print('---------------------------------------------------------------------------')
    print('')
    print('Check results in: ', OutDir)