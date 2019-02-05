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
import pandas
import time
import warnings
import subprocess
from BLM.blm_eval import blm_eval
np.set_printoptions(threshold=np.nan)

# Developer notes:
# --------------------------------------------------------------------------
# In the following code I have used the following subscripts to indicate:
#
# _r - This means this is an array of values corresponding to voxels which
#      are present in between k and n_s-1 studies (inclusive), where k is
#      decided by the user specified thresholds. These voxels will typically
#      be on the edge of the brain and look like a "ring" around the brain,
#      hence "_r" for ring.
# 
# _i - This means that this is an array of values corresponding to voxels 
#      which are present in all n_s studies. These will usually look like
#      a smaller mask place inside the whole study mask. Hence "_i" for 
#      inner.
#
# _sv - This means this variable is spatially varying (There is a reading
#       per voxel). 
#
# --------------------------------------------------------------------------
# Author: Tom Maullin (04/02/2019)

def main(*args):

    # ----------------------------------------------------------------------
    # Check inputs
    # ----------------------------------------------------------------------
    if len(args)==0:
        # Load in inputs
        with open(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..',
                    'blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        # In this case inputs is first argument
        inputs = args[0]

    # ----------------------------------------------------------------------
    # Read basic inputs
    # ----------------------------------------------------------------------
    OutDir = inputs['outdir']
    
    # Get number of parameters
    c1 = blm_eval(inputs['contrasts'][0]['c' + str(1)]['vector'])
    c1 = np.array(c1)
    n_p = c1.shape[0]
    del c1
    
    # Read in the nifti size and work out number of voxels.
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = nib.load(nifti_path)

    NIFTIsize = nifti.shape
    n_v = int(np.prod(NIFTIsize))

    # ----------------------------------------------------------------------
    # Load X'X, X'Y, Y'Y and n_s
    # ----------------------------------------------------------------------
    if len(args)==0:
        # Read the matrices from the first batch. Note XtY is transposed as pandas
        # handles lots of rows much faster than lots of columns.
        sumXtX = pandas.io.parsers.read_csv(os.path.join(OutDir,"tmp","XtX1.csv"), 
                            sep=",", header=None).values
        sumXtY = pandas.io.parsers.read_csv(os.path.join(OutDir,"tmp","XtY1.csv"), 
                            sep=",", header=None).values.transpose()
        sumYtY = pandas.io.parsers.read_csv(os.path.join(OutDir,"tmp","YtY1.csv"), 
                            sep=",", header=None).values
        nmapb  = nib.load(os.path.join(OutDir,"tmp", "blm_vox_n_batch1.nii"))
        n_s_sv = nmapb.get_data()

        # Delete the files as they are no longer needed.
        os.remove(os.path.join(OutDir,"tmp","XtX1.csv"))
        os.remove(os.path.join(OutDir,"tmp","XtY1.csv"))
        os.remove(os.path.join(OutDir,"tmp","YtY1.csv"))
        os.remove(os.path.join(OutDir,"tmp","blm_vox_n_batch1.nii"))

        # Work out how many files we need.
        XtX_files = glob.glob(os.path.join(OutDir,"tmp","XtX*"))

        # Cycle through batches and add together results.
        for batchNo in range(2,(len(XtX_files)+2)):

            # Sum the batches.
            sumXtX = sumXtX + pandas.io.parsers.read_csv(
                os.path.join(OutDir,"tmp","XtX" + str(batchNo) + ".csv"), 
                             sep=",", header=None).values

            sumXtY = sumXtY + pandas.io.parsers.read_csv(
                os.path.join(OutDir,"tmp","XtY" + str(batchNo) + ".csv"), 
                             sep=",", header=None).values.transpose()

            sumYtY = sumYtY + pandas.io.parsers.read_csv(
                os.path.join(OutDir,"tmp","YtY" + str(batchNo) + ".csv"), 
                             sep=",", header=None).values
            
            # Obtain the full nmap.
            n_s_sv = n_s_sv + nib.load(os.path.join(OutDir,"tmp", 
                "blm_vox_n_batch" + str(batchNo) + ".nii")).get_data()
            
            # Delete the files as they are no longer needed.
            os.remove(os.path.join(OutDir, "tmp","XtX" + str(batchNo) + ".csv"))
            os.remove(os.path.join(OutDir, "tmp","XtY" + str(batchNo) + ".csv"))
            os.remove(os.path.join(OutDir, "tmp","YtY" + str(batchNo) + ".csv"))
            os.remove(os.path.join(OutDir, "tmp", "blm_vox_n_batch" + str(batchNo) + ".nii"))

    else:
        # Read in sums.
        sumXtX = args[1]
        sumXtY = args[2].transpose()
        sumYtY = args[3]
        n_s_sv = args[4]

    # Save nmap
    nmap = nib.Nifti1Image(n_s_sv,
                           nifti.affine,
                           header=nifti.header)
    nib.save(nmap, os.path.join(OutDir,'blm_vox_n.nii'))
    n_s_sv = n_s_sv.reshape(n_v, 1)
    del nmap

    # Dimension bug handling
    if np.ndim(sumXtX) == 0:
        sumXtX = np.array([[sumXtX]])
    elif np.ndim(sumXtX) == 1:
        sumXtX = np.array([sumXtX])

    if np.ndim(sumXtY) == 0:
        sumXtY = np.array([[sumXtY]])
    elif np.ndim(sumXtY) == 1:
        sumXtY = np.array([sumXtY])

    # Get ns.
    X = pandas.io.parsers.read_csv(
        inputs['X'], sep=',', header=None).values
    n_s = X.shape[0]

    # ----------------------------------------------------------------------
    # Create Mask
    # ----------------------------------------------------------------------

    Mask = np.ones([n_v, 1])

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
            Mask[n_s_sv<rmThresh*n_s]=0

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
            Mask[n_s_sv<amThresh]=0

        if ("Masking" in inputs["Missingness"]) or ("masking" in inputs["Missingness"]):

            # Read in threshold mask
            if "Masking" in inputs["Missingness"]:
                mmThresh_path = inputs["Missingness"]["Masking"]
            else:
                mmThresh_path = inputs["Missingness"]["masking"]

            try:
                # Read in the mask nifti.
                mmThresh = nib.load(mmThresh_path)

                # Check whether the mask has the same shape as the other niftis
                if np.array_equal(mmThresh.shape, NIFTIsize):
                    mmThresh = mmThresh.get_data().reshape([n_v, 1])
                else:
                    # Make flirt resample command
                    resamplecmd = ["flirt", "-in", mmThresh_path,
                                            "-ref", nifti_path,
                                            "-out", os.path.join(OutDir, 'tmp', 'blm_im_resized.nii'),
                                            "-applyxfm"]

                    # Warn the user about what is happening.
                    warnings.warn('Masking NIFTI ' + mmThresh_path + ' does not have the'\
                                  ' same dimensions as the input data and will therefore'\
                                  ' be resampled using FLIRT.')

                    # Run the command
                    process = subprocess.Popen(resamplecmd, shell=False,
                                               stdout=subprocess.PIPE)
                    out, err = process.communicate()

                    # Check the NIFTI has been generate, else wait up to 5 minutes.
                    t = time.time()
                    t1 = 0
                    while (not os.path.isfile(os.path.join(OutDir, 'tmp', 'blm_im_resized.nii.gz'))) and (t1 < 300):
                        t1 = time.time() - t

                    # Load the newly resized nifti mask
                    mmThresh = nib.load(os.path.join(OutDir, 'tmp', 'blm_im_resized.nii.gz'))

                    mmThresh = mmThresh.get_data().reshape([n_v, 1])

            except:
                raise ValueError('Nifti image ' + mmThresh_path + ' will not load.')

            # Apply mask nifti.
            Mask[mmThresh==0]=0


    # We remove anything with 1 degree of freedom (or less) by default.
    # 1 degree of freedom seems to cause broadcasting errors on a very
    # small percentage of voxels.
    Mask[n_s_sv<=n_p+1]=0

    # Reshape sumXtX to correct n_v by n_p by n_p
    sumXtX = sumXtX.reshape([n_v, n_p, n_p])

    # We also remove all voxels where the design has a column of just
    # zeros.
    for i in range(0,n_p):
        Mask[np.where(sumXtX[:,i,i]==0)]=0

    # Remove voxels with designs without full rank.
    M_inds = np.where(Mask==1)[0]
    Mask[M_inds[np.where(
        np.absolute(blm_det(sumXtX[M_inds,:,:])) < np.sqrt(sys.float_info.epsilon)
        )]]=0

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

    # Get indices of voxels in ring around brain where there are
    # missing studies.
    R_inds = np.where((Mask==1)*(n_s_sv<n_s))[0]

    # Get indices of the "inner" volume where all studies had information
    # present. I.e. the voxels (usually near the middle of the brain) where
    # every voxel has a reading for every study.
    I_inds = np.where((Mask==1)*(n_s_sv==n_s))[0]
    del Mask

    # Number of voxels in ring
    n_v_r = R_inds.shape[0]

    # Number of voxels in inner mask
    n_v_i = I_inds.shape[0]

    # ----------------------------------------------------------------------
    # Calculate betahat = (X'X)^(-1)X'Y and output beta maps
    # ----------------------------------------------------------------------    

    # Calculate masked X'X for ring
    sumXtX_r = sumXtX[R_inds,:,:]

    # Calculate masked X'Y for ring
    sumXtY = sumXtY.transpose()
    sumXtY = sumXtY.reshape([n_v, n_p, 1])
    sumXtY_r = sumXtY[R_inds,:]

    # Calculate masked Beta for ring
    beta_r = np.linalg.solve(sumXtX_r, sumXtY_r)

    # Unmask Beta
    beta = np.zeros([n_v, n_p])

    # Outer ring values
    beta[R_inds,:] = beta_r.reshape([n_v_r, n_p])


    # If we have indices where all studies are present, work out X'X and
    # X'Y for these studies.
    if I_inds.size!=0:
        
        # X'X must be 1 by np by np for broadcasting
        sumXtX_i = sumXtX[I_inds[0],:,:]
        sumXtX_i = sumXtX_i.reshape([1, n_p, n_p])

        sumXtY_i = sumXtY[I_inds,:]

        # Calculate beta
        beta_i = np.linalg.solve(sumXtX_i, sumXtY_i)
        beta[I_inds,:] = beta_i.reshape([n_v_i,n_p])

    beta = beta.reshape([n_v, n_p]).transpose()


    # Cycle through betas and output results.
    for k in range(0,beta.shape[0]):

        betak = beta[k,:].reshape(int(NIFTIsize[0]),
                                  int(NIFTIsize[1]),
                                  int(NIFTIsize[2]))

        # Save beta map.
        betakmap = nib.Nifti1Image(betak,
                                   nifti.affine,
                                   header=nifti.header)
        nib.save(betakmap, os.path.join(OutDir,'blm_vox_beta_b' + str(k+1) + '.nii'))
        del betak, betakmap

    del sumXtY, sumXtY_r, sumXtX, sumXtY_i

    if np.ndim(beta) == 0:
        beta = np.array([[beta]])
    elif np.ndim(beta) == 1:
        beta = np.array([beta])

    # ----------------------------------------------------------------------
    # Calculate residual sum of squares e'e = Y'Y - (Xb)'Xb
    # ---------------------------------------------------------------------- 

    # Reshape beta along smallest axis for quicker
    # residual calculation
    beta_r_t = beta_r.transpose(0,2,1)
    beta_i_t = beta_i.transpose(0,2,1)

    # Calculate Beta transpose times XtX and delete the
    # now redudundant matrices.
    betatXtX_r = np.matmul(beta_r_t, sumXtX_r)
    betatXtX_i = np.matmul(beta_i_t, sumXtX_i)
    del beta_r_t, beta_i_t

    # Multiply BetatXtX by Beta and delete the reduundant
    # matrices.
    betatXtXbeta_r = np.matmul(betatXtX_r, beta_r)
    betatXtXbeta_i = np.matmul(betatXtX_i, beta_i)
    del betatXtX_r, betatXtX_i

    # Reshape betat XtX beta
    betatXtXbeta_r = np.reshape(betatXtXbeta_r, [n_v_r,1])
    betatXtXbeta_i = np.reshape(betatXtXbeta_i, [n_v_i,1])

    # Residual sum of squares
    ete_r = sumYtY[R_inds] - betatXtXbeta_r
    ete_i = sumYtY[I_inds] - betatXtXbeta_i

    del sumYtY, betatXtXbeta_r, betatXtXbeta_i

    # ----------------------------------------------------------------------
    # Calculate residual mean squares = e'e/(n_s - n_p)
    # ----------------------------------------------------------------------

    # Mask spatially varying n_s
    n_s_sv_r = n_s_sv[R_inds,:]

    # In spatially varying the degrees of freedom
    # varies across voxels
    resms_r = ete_r/(n_s_sv_r-n_p)
    resms_i = ete_i/(n_s-n_p)

    # Unmask resms
    resms = np.zeros([n_v,1])
    resms[R_inds,:] = resms_r
    resms[I_inds,:] = resms_i
    resms = resms.reshape(NIFTIsize[0], 
                          NIFTIsize[1],
                          NIFTIsize[2])

    # Output ResSS.
    msmap = nib.Nifti1Image(resms,
                            nifti.affine,
                            header=nifti.header)
    nib.save(msmap, os.path.join(OutDir,'blm_vox_resms.nii'))
    del msmap, resms

    # ----------------------------------------------------------------------
    # Calculate beta covariance maps
    # ----------------------------------------------------------------------
        
    # Calculate masked (x'X)^(-1) values for ring
    isumXtX_r = blm_inverse(sumXtX_r, ouflow=True)
    isumXtX_i = blm_inverse(sumXtX_i, ouflow=True)

    # Output variance for each pair of betas
    for i in range(0,n_p):
        for j in range(0,n_p):

                # Calculate masked cov beta ij for ring
                covbetaij_r = np.multiply(
                    resms_r.reshape([resms_r.shape[0]]),
                    isumXtX_r[:,i,j])
                covbetaij_i = np.multiply(
                    resms_i.reshape([resms_i.shape[0]]),
                    isumXtX_i[:,i,j])

                # Unmask cov beta ij
                covbetaij = np.zeros([n_v])
                covbetaij[R_inds] = covbetaij_r
                covbetaij[I_inds] = covbetaij_i
                covbetaij = covbetaij.reshape(
                                        NIFTIsize[0],
                                        NIFTIsize[1],
                                        NIFTIsize[2],
                                        )
                    
                # Output covariance map
                covbetaijmap = nib.Nifti1Image(covbetaij,
                                               nifti.affine,
                                               header=nifti.header)
                nib.save(covbetaijmap,
                    os.path.join(OutDir, 
                        'blm_vox_cov_b' + str(i+1) + ',' + str(j+1) + '.nii'))
                del covbetaij, covbetaijmap

    # ----------------------------------------------------------------------
    # Calculate COPEs, statistic maps and covariance maps.
    # ----------------------------------------------------------------------
    n_c = len(inputs['contrasts'])

    for i in range(0,n_c):

        # Read in contrast vector
        # Get number of parameters
        cvec = blm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        cvec = np.array(cvec)

        # Calculate C\hat{\beta}}
        cbeta_r = np.matmul(cvec, beta_r)
        cbeta_i = np.matmul(cvec, beta_i)
        if cvec.ndim == 1:
            cvec = cvec.reshape([1,cvec.shape[0]])

        if inputs['contrasts'][i]['c' + str(i+1)]['statType'] == 'T':

            # A T contrast has only one row so we can output cbeta here
            cbeta = np.zeros([n_v,1])
            cbeta[R_inds,:] = cbeta_r
            cbeta[I_inds,:] = cbeta_i
            cbeta = cbeta.reshape(
                        NIFTIsize[0],
                        NIFTIsize[1],
                        NIFTIsize[2],
                        )

            # Output cbeta/cope map
            cbetamap = nib.Nifti1Image(cbeta,
                                       nifti.affine,
                                       header=nifti.header)
            nib.save(cbetamap,
                os.path.join(OutDir, 
                    'blm_vox_beta_c' + str(i+1) + '.nii'))
            del cbeta, cbetamap

            # Calculate c'(X'X)^(-1)c
            cvectiXtXcvec_r = np.matmul(
                np.matmul(cvec, isumXtX_r),
                np.transpose(cvec)).reshape(n_v_r)
            cvectiXtXcvec_i = np.matmul(
                np.matmul(cvec, isumXtX_i),
                np.transpose(cvec))

            # Calculate masked cov(c\hat{\beta}) for ring
            covcbeta_r = cvectiXtXcvec_r*resms_r.reshape(n_v_r)

            # Calculate masked cov(c\hat{\beta}) for inner
            covcbeta_i = cvectiXtXcvec_i*resms_i.reshape(n_v_i)

            # Unmask to output
            covcbeta = np.zeros([n_v])
            covcbeta[R_inds] = covcbeta_r
            covcbeta[I_inds] = covcbeta_i
            covcbeta = covcbeta.reshape(
                NIFTIsize[0],
                NIFTIsize[1],
                NIFTIsize[2]
                )

            # Output covariance map
            covcbetamap = nib.Nifti1Image(covcbeta,
                                          nifti.affine,
                                          header=nifti.header)
            nib.save(covcbetamap,
                os.path.join(OutDir, 
                    'blm_vox_cov_c' + str(i+1) + '.nii'))
            del covcbeta, covcbetamap

            # Calculate masked T statistic image for ring
            tStatc_r = cbeta_r.reshape(n_v_r)/np.sqrt(covcbeta_r)
            tStatc_i = cbeta_i.reshape(n_v_i)/np.sqrt(covcbeta_i)

            # Unmask T stat
            tStatc = np.zeros([n_v])
            tStatc[R_inds] = tStatc_r
            tStatc[I_inds] = tStatc_i
            tStatc = tStatc.reshape(
                NIFTIsize[0],
                NIFTIsize[1],
                NIFTIsize[2]
                )

            # Output statistic map
            tStatcmap = nib.Nifti1Image(tStatc,
                                        nifti.affine,
                                        header=nifti.header)
            nib.save(tStatcmap,
                os.path.join(OutDir, 
                    'blm_vox_Tstat_c' + str(i+1) + '.nii'))
            del tStatc, tStatcmap, tStatc_i, tStatc_r

        if inputs['contrasts'][i]['c' + str(i+1)]['statType'] == 'F':
                
            # Get dimension of cvector
            q = cvec.shape[0]

            # Calculate c'(X'X)^(-1)c
            # (Note C is read in the other way around for F)
            cvectiXtXcvec_r = np.matmul(
                np.matmul(cvec, isumXtX_r),
                np.transpose(cvec))
            cvectiXtXcvec_i = np.matmul(
                np.matmul(cvec, isumXtX_i),
                np.transpose(cvec))

            # Cbeta needs to be nvox by 1 by npar for stacked
            # multiply.
            cbetat_r = cbeta_r.transpose(0,2,1)
            cbetat_i = cbeta_i.transpose(0,2,1)

            # Calculate masked (c'(X'X)^(-1)c)^(-1) values for ring
            icvectiXtXcvec_r = blm_inverse(cvectiXtXcvec_r, ouflow=True).reshape([n_v_r, q*q])
            icvectiXtXcvec_i = blm_inverse(cvectiXtXcvec_i, ouflow=True).reshape([1, q*q])

            # Make (c'(X'X)^(-1)c)^(-1) unmasked
            icvectiXtXcvec = np.zeros([n_v, q*q])
            icvectiXtXcvec[R_inds,:]=icvectiXtXcvec_r
            icvectiXtXcvec[I_inds,:]=icvectiXtXcvec_i
            icvectiXtXcvec = icvectiXtXcvec.reshape([n_v, q, q])


            # Calculate the numerator of the F statistic for the ring
            Fnumerator_r = np.matmul(
                cbetat_r,
                np.linalg.solve(cvectiXtXcvec_r, cbeta_r))
            # Calculate the numerator of the F statistic for the inner 
            Fnumerator_i = np.matmul(
                cbetat_i,
                np.linalg.solve(cvectiXtXcvec_i, cbeta_i))

            Fnumerator_r = Fnumerator_r.reshape(Fnumerator_r.shape[0])
            Fnumerator_i = Fnumerator_i.reshape(Fnumerator_i.shape[0])

            # Calculate the denominator of the F statistic for ring
            Fdenominator_r = q*resms_r.reshape([n_v_r])

            # Calculate the denominator of the F statistic for inner
            Fdenominator_i = q*resms_i.reshape([n_v_i])

            # Calculate F statistic.
            fStatc_r = Fnumerator_r/Fdenominator_r
            fStatc_i = Fnumerator_i/Fdenominator_i

            # Save F statistic
            fStatc = np.zeros([n_v])
            fStatc[R_inds]=fStatc_r
            fStatc[I_inds]=fStatc_i
            fStatc = fStatc.reshape(
                               NIFTIsize[0],
                               NIFTIsize[1],
                               NIFTIsize[2]
                           )

            # Output statistic map
            fStatcmap = nib.Nifti1Image(fStatc,
                                        nifti.affine,
                                        header=nifti.header)
            nib.save(fStatcmap,
                os.path.join(OutDir, 
                    'blm_vox_Fstat_c' + str(i+1) + '.nii'))
            del fStatc, fStatcmap

            # Mask spatially varying n_s
            n_s_sv_r = n_s_sv_r.reshape([n_v_r])

            # Calculate partial R2 masked for ring.
            partialR2_r = (q*fStatc_r)/(q*fStatc_r + n_s_sv_r - n_p)
            partialR2_i = (q*fStatc_i)/(q*fStatc_i + n_s - n_p)

            # Unmask partialR2.
            partialR2 = np.zeros([n_v])
            partialR2[R_inds] = partialR2_r
            partialR2[I_inds] = partialR2_i

            partialR2 = partialR2.reshape(
                               NIFTIsize[0],
                               NIFTIsize[1],
                               NIFTIsize[2]
                           )

            # Output statistic map
            partialR2map = nib.Nifti1Image(partialR2,
                                        nifti.affine,
                                        header=nifti.header)
            nib.save(partialR2map,
                os.path.join(OutDir, 
                    'blm_vox_pr2_c' + str(i+1) + '.nii'))
            del partialR2, partialR2map


    # Clean up files
    if len(args)==0:
        os.remove(os.path.join(OutDir, 'nb.txt'))
    shutil.rmtree(os.path.join(OutDir, 'tmp'))

    w.resetwarnings()


# This function inverts matrix A. If ouflow is True,
# special handling is used to account for over/under
# flow. In this case, it assumes that A has non-zero
# diagonals.
def blm_inverse(A, ouflow=False):


    # If ouflow is true, we need to precondition A.
    if ouflow:

        # Work out number of matrices and dimension of
        # matrices. I.e. if we have seven 3 by 3 matrices
        # to invert n_r = 7, d_r = 3.
        n_r = A.shape[0]
        d_r = A.shape[1]

        # Make D to be filled with diagonal elements
        D = np.broadcast_to(np.eye(d_r), (n_r,d_r,d_r)).copy()

        # Obtain 1/sqrt(diagA)
        diagA = 1/np.sqrt(A.diagonal(0,1,2))
        diagA = diagA.reshape(n_r, d_r)

        # Make this back into diagonal matrices
        diaginds = np.diag_indices(d_r)
        D[:, diaginds[0], diaginds[1]] = diagA 

        # Precondition A.
        A = np.matmul(np.matmul(D, A), D)

    # np linalg inverse doesn't handle dim=[1,1]
    if np.ndim(A) == 1:
        iA = 1/A
    else:
        iA = np.linalg.solve(A, np.eye(d_r).reshape(1,d_r,d_r))

    if ouflow:

        # Undo preconditioning.
        iA = np.matmul(np.matmul(D, iA), D)

    return(iA)

# This function calculates the determinant of matrix A/
# stack of matrices A, with special handling accounting
# for over/under flow. 
def blm_det(A):


    # Precondition A.
    # Work out number of matrices and dimension of
    # matrices. I.e. if we have seven 3 by 3 matrices
    # to invert n_r = 7, d_r = 3.
    n_r = A.shape[0]
    d_r = A.shape[1]

    # Make D to be filled with diagonal elements
    D = np.broadcast_to(np.eye(d_r), (n_r,d_r,d_r)).copy()

    # Obtain 1/sqrt(diagA)
    diagA = 1/np.sqrt(A.diagonal(0,1,2))
    diagA = diagA.reshape(n_r, d_r)

    # Make this back into diagonal matrices
    diaginds = np.diag_indices(d_r)
    D[:, diaginds[0], diaginds[1]] = diagA 

    # Calculate DAD.
    DAD = np.matmul(np.matmul(D, A), D)

    # Calculate determinants.
    detDAD = np.linalg.det(DAD)
    detDD = np.prod(diagA, axis=1)
    
    # Calculate determinant of A
    detA = detDAD/detDD

    return(detA)

if __name__ == "__rain__":
    main()
