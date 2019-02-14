import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import warnings
import resource
import nibabel as nib
import sys
import os
import shutil
import yaml
from BLM.blm_eval import blm_eval

def main(*args):

    # Change to blm directory
    pwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if len(args)==0 or (not args[0]):
        # Load in inputs
        ipath = os.path.abspath(os.path.join('..','blm_config.yml'))
        with open(ipath, 'r') as stream:
            inputs = yaml.load(stream)
    else:
        if type(args[0]) is str:
            ipath = os.path.abspath(os.path.join(pwd, args[0]))
            # In this case inputs file is first argument
            with ipath as stream:
                inputs = yaml.load(stream)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[0]
            ipath = ''

    print(ipath)

    # Save absolute filepaths in place of relative filepaths
    if ipath: 

        # Y files
        if not os.path.isabs(inputs['Y_files']):

            # Change Y in inputs
            inputs['Y_files'] = os.path.join(pwd, inputs['Y_files'])

        # If mask files are specified
        if 'M_files' in inputs:

            # M_files
            if not os.path.isabs(inputs['M_files']):

                # Change M in inputs
                inputs['M_files'] = os.path.join(pwd, inputs['M_files'])

        # If X is specified
        if not os.path.isabs(inputs['X']):

            # Change X in inputs
            inputs['X'] = os.path.join(pwd, inputs['X'])

        if not os.path.isabs(inputs['outdir']):

            # Change output directory in inputs
            inputs['outdir'] = os.path.join(pwd, inputs['outdir'])

        # Change missingness mask in inputs
        if 'Missingness' in inputs:

            if ("Masking" in inputs["Missingness"]) or ("masking" in inputs["Missingness"]):

                # Read in threshold mask
                if not os.path.isabs(inputs["Missingness"]["Masking"]):
                    if "Masking" in inputs["Missingness"]:
                        inputs["Missingness"]["Masking"] = os.path.join(pwd, inputs["Missingness"]["Masking"])

                if not os.path.isabs(inputs["Missingness"]["masking"]):
                    if "Masking" in inputs["Missingness"]:
                        inputs["Missingness"]["masking"] = os.path.join(pwd, inputs["Missingness"]["masking"])

        # Update inputs
        with open(ipath, 'w') as outfile:
            yaml.dump(inputs, outfile, default_flow_style=False)


    print(os.path.isabs(inputs['Y_files']))

    # Change paths to absoluate if they aren't already
    
    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    OutDir = inputs['outdir']

    # Get number of parameters
    c1 = blm_eval(inputs['contrasts'][0]['c' + str(1)]['vector'])
    c1 = np.array(c1)
    n_p = c1.shape[0]
    del c1

    # Make output directory and tmp
    if not os.path.isdir(OutDir):
        os.mkdir(OutDir)
    if not os.path.isdir(os.path.join(OutDir, "tmp")):
        os.mkdir(os.path.join(OutDir, "tmp"))

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

    # Get the maximum memory a NIFTI could take in storage. We divide by 3
    # as approximately a third of the volume is actually non-zero/brain
    NIFTIsize = sys.getsizeof(np.zeros(Y0.shape,dtype='uint64'))

    if NIFTIsize > MAXMEM:
        raise ValueError('The NIFTI "' + Y_files[0] + '"is too large')

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use. We divide NIFTIsize by 3
    # as approximately a third of the volume is actually non-zero/brain 
    # and then also divide though everything by the number of parameters
    # in the analysis.
    blksize = np.floor(MAXMEM/8/NIFTIsize/n_p);
    if blksize == 0:
        raise ValueError('Blocksize too small.')

    # Check F contrast ranks 
    n_c = len(inputs['contrasts'])
    for i in range(0,n_c):

        if inputs['contrasts'][i]['c' + str(i+1)]['statType'] == 'F':

            # Read in contrast vector
            # Get number of parameters
            cvec = blm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
            cvec = np.array(cvec)
                
            # Get dimension of cvector
            q = cvec.shape[0]

            if np.linalg.matrix_rank(cvec)<q:
                raise ValueError('F contrast: \n' + str(cvec) + '\n is not of correct rank.')

    if (len(args)==0) or (type(args[0]) is str):
        with open(os.path.join(OutDir, "nb.txt"), 'w') as f:
            print(int(np.ceil(len(Y_files)/int(blksize))), file=f)
    else:
        return(int(np.ceil(len(Y_files)/int(blksize))))

    w.resetwarnings()

if __name__ == "__main__":
    main()
