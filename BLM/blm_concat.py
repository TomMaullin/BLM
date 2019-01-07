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
np.set_printoptions(threshold=np.nan)

def main(*args):

    # ----------------------------------------------------------------------
    # Check inputs
    # ----------------------------------------------------------------------
    if len(args)==0:
        # Load in inputs
        with open(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..',
                    'blm_defaults.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        # In this case inputs is first argument
        inputs = args[0]

    # ----------------------------------------------------------------------
    # Read basic inputs
    # ----------------------------------------------------------------------
    SVFlag = inputs['SVFlag']
    OutDir = inputs['outdir']
    
    # Get number of parameters
    c1 = np.array(inputs['contrasts'][0]['c' + str(1)]['vector'])
    n_p = c1.shape[0]
    print(n_p)
    del c1
    
    # Read in the nifti size and work out number of voxels.
    with open(inputs['Y_files']) as a:
        nifti = nib.load(a.readline().replace('\n', ''))

    NIFTIsize = nifti.shape
    n_v = int(n_v)

    # ----------------------------------------------------------------------
    # Load X'X, X'Y, Y'Y and n_s
    # ----------------------------------------------------------------------

    # Read the matrices from the first batch. Note XtY is transposed as pandas
    # handles lots of rows much faster than lots of columns.
    sumXtX = pandas.io.parsers.read_csv(os.path.join(OutDir,"tmp","XtX1.csv"), 
                        sep=",", header=None).values
    sumXtY = pandas.io.parsers.read_csv(os.path.join(OutDir,"tmp","XtY1.csv"), 
                        sep=",", header=None).values.transpose()
    sumYtY = pandas.io.parsers.read_csv(os.path.join(OutDir,"tmp","YtY1.csv"), 
                        sep=",", header=None).values
    nmapb  = nib.load(os.path.join(OutDir,"tmp", "blm_vox_n_batch1.nii"))
    nmapd = nmapb.get_data()

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
        nmapd = nmapd + nib.load(os.path.join(OutDir,"tmp", 
            "blm_vox_n_batch" + str(batchNo) + ".nii")).get_data()
        
        # Delete the files as they are no longer needed.
        os.remove(os.path.join(OutDir, "tmp","XtX" + str(batchNo) + ".csv"))
        os.remove(os.path.join(OutDir, "tmp","XtY" + str(batchNo) + ".csv"))
        os.remove(os.path.join(OutDir, "tmp","YtY" + str(batchNo) + ".csv"))
        os.remove(os.path.join(OutDir, "tmp", "blm_vox_n_batch" + str(batchNo) + ".nii"))

    # Output final n map
    nmap = nib.Nifti1Image(nmapd,
                           nmapb.affine,
                           header=nmapb.header)
    nib.save(nmap, os.path.join(OutDir,'blm_vox_n.nii'))

    # Dimension bug handling
    if np.ndim(sumXtX) == 0:
        sumXtX = np.array([[sumXtX]])
    elif np.ndim(sumXtX) == 1:
        sumXtX = np.array([sumXtX])

    if np.ndim(sumXtY) == 0:
        sumXtY = np.array([[sumXtY]])
    elif np.ndim(sumXtY) == 1:
        sumXtY = np.array([sumXtY])

    # ----------------------------------------------------------------------
    # Create Mask
    # ----------------------------------------------------------------------

    Mask = np.zeros([n_v, 1])
    # Add all voxels where n_s > n_p
    n_s = nib.load(os.path.join(OutDir,'blm_vox_n.nii'))
    n_s = n_s.get_data()
    print(Mask.shape)

    # If spatially varying remove the designs that aren't of full rank.
    if SVFlag:

        Mask[n_s.reshape(n_v, 1)>n_p]=1

        # Reshape sumXtX to correct n_v by n_p by n_p
        sumXtX = sumXtX.reshape([n_v, n_p, n_p])

        # Remove voxels with designs without full rank.
        Mask[np.where(np.linalg.det(sumXtX)==0)[0]]=0
        print(np.where(np.linalg.det(sumXtX)==0)[0])

    else:

        Mask[n_s.reshape(n_v, 1) > 1] = 1

    # Output final mask map
    maskmap = nib.Nifti1Image(Mask.reshape(
                                    NIFTIsize[0],
                                    NIFTIsize[1],
                                    NIFTIsize[2]
                                    ),
                              nmapb.affine,
                              header=nmapb.header)
    nib.save(maskmap, os.path.join(OutDir,'blm_vox_mask.nii'))

    # Get indices of voxels in mask.
    M_inds = np.where(Mask==1)[0]

    # Number of voxels in mask
    n_v_m = M_inds.shape[0]

    # ----------------------------------------------------------------------
    # Calculate (X'X)^(-1)
    # ----------------------------------------------------------------------
    # Mask and reshape if we are using a spatially varying design.
    if SVFlag:
        
        # Calculate masked (x'X)^(-1) values
        sumXtX_m = sumXtX[M_inds,:,:]
        isumXtX_m = np.linalg.inv(sumXtX_m).reshape([n_v_m, n_s*n_s])

        # Make (X'X)^(-1) unmasked
        isumXtX = np.zeros([n_v, n_s*n_s])
        isumXtX[M_inds,:]=isumXtX_m
        isumXtX = isumXtX.reshape([n_v, n_s, n_s])


    # If we are not using a spatially varying design, inverse in
    # the normal manner.
    else:
        # Calculate inverse of XtX
        isumXtX = blm_inverse(sumXtX)

    # If we are doing spatially varying we need to reshape XtY.
    if SVFlag:
        sumXtY = sumXtY.transpose()
        sumXtY = sumXtY.reshape([n_v, n_p, 1])
    else:
        # If we are doing non-spatially varying we still need to mask XtY
        sumXtY[:, np.where(Mask==0)]=0

    # ----------------------------------------------------------------------
    # Calculate betahat = (X'X)^(-1)X'Y and output beta maps
    # ----------------------------------------------------------------------    

    beta = np.matmul(isumXtX, sumXtY)

    if SVFlag:
        beta = beta.reshape([beta.shape[0], beta.shape[1]]).transpose()

    # Cycle through betas and output results.
    for i in range(0,beta.shape[0]):

        betai = beta[i,:].reshape(int(NIFTIsize[0]),
                                  int(NIFTIsize[1]),
                                  int(NIFTIsize[2]))

        # Save beta map.
        betaimap = nib.Nifti1Image(betai,
                                   nifti.affine,
                                   header=nifti.header)
        nib.save(betaimap, os.path.join(OutDir,'blm_vox_beta_b' + str(i+1) + '.nii'))

    del betai, betaimap

    if np.ndim(beta) == 0:
        beta = np.array([[beta]])
    elif np.ndim(beta) == 1:
        beta = np.array([beta])

    # ----------------------------------------------------------------------
    # Calculate residual sum of squares e'e = Y'Y - (Xb)'Xb
    # ---------------------------------------------------------------------- 

    # Reshape beta along smallest axis for quicker
    # residual calculation
    beta_rs = np.zeros([beta.shape[1], beta.shape[0], 1])
    beta_rs_t = np.zeros([beta.shape[1], 1, beta.shape[0]])
    for i in range(0,beta.shape[0]):

       beta_rs[:, i, 0] = beta[i,:]
       beta_rs_t[:, 0, i] = beta[i,:]

    # Calculate Beta transpose times XtX and delete the
    # now redudundant matrices.
    betatXtX = np.matmul(beta_rs_t, sumXtX)
    del beta_rs_t

    # Multiply BetatXtX by Beta and delete the reduundant
    # matrices.
    betatXtXbeta = np.matmul(betatXtX, beta_rs)
    del betatXtX, beta_rs

    # Reshape betat XtX beta
    betatXtXbeta = np.reshape(betatXtXbeta, [betatXtXbeta.shape[0],1])

    # Residual sum of squares
    ete_m = sumYtY[M_inds] - betatXtXbeta[M_inds]

    # Unmask ete
    ete = np.zeros([n_v, 1])
    ete[M_inds]=ete_m
    ete = ete.reshape(int(NIFTIsize[0]),
                      int(NIFTIsize[1]),
                      int(NIFTIsize[2]))

    # ----------------------------------------------------------------------
    # Calculate residual mean squares = e'e/(n_s - n_p)
    # ----------------------------------------------------------------------

    # Get residual mean squares by dividing by degrees of
    # freedom
    if not SVFlag:

        # Get number of scans and number of parameters
        X = pandas.io.parsers.read_csv(
            inputs['X'], sep=',', header=None).values
        n_s = X.shape[0]

        # In non spatially varying the degrees of freedom
        # are fixed across voxels
        resms = ete/(n_s-n_p)

    else:

        # Mask n_s
        n_s_m = n_s.reshape(n_v, 1)
        n_s_m = n_s_m[M_inds,:]

        # Mask ete
        ete_m = ete.reshape(n_v, 1)
        ete_m = ete_m[M_inds,:]

        # In spatially varying the degrees of freedom
        # varies across voxels
        resms_m = ete_m/(n_s_m-n_p)

        # Unmask resms
        resms = np.zeros([n_v,1])
        resms[M_inds,:] = resms_m
        resms = resms.reshape(NIFTIsize[0], 
                              NIFTIsize[1],
                              NIFTIsize[2])

    # Output ResSS.
    msmap = nib.Nifti1Image(resms,
                            nifti.affine,
                            header=nifti.header)
    nib.save(msmap, os.path.join(OutDir,'blm_vox_resms.nii'))

    # calculate beta covariance maps
    if not SVFlag:

        # Output variance for each pair of betas
        for i in range(0,isumXtX.shape[0]):
            for j in range(0,isumXtX.shape[1]):

                    # Calculate covariance of beta i and beta j.
                    covbetaij = resms*isumXtX[i,j]

                    # Output covariance map
                    covbetaijmap = nib.Nifti1Image(covbetaij,
                                                   nifti.affine,
                                                   header=nifti.header)
                    nib.save(covbetaijmap,
                        os.path.join(OutDir, 
                            'blm_vox_cov_b' + str(i+1) + ',' + str(j+1) + '.nii'))

        del covbetaijmap

    else:

        # Output variance for each pair of betas
        for i in range(0,isumXtX.shape[1]):
            for j in range(0,isumXtX.shape[2]):

                    covbetaij = np.multiply(resms,
                        isumXtX[:,i,j].reshape(
                            resms.shape[0],
                            resms.shape[1],
                            resms.shape[2],
                            ))
                        
                    # Output covariance map
                    covbetaijmap = nib.Nifti1Image(covbetaij,
                                                   nifti.affine,
                                                   header=nifti.header)
                    nib.save(covbetaijmap,
                        os.path.join(OutDir, 
                            'blm_vox_cov_b' + str(i+1) + ',' + str(j+1) + '.nii'))

        del covbetaijmap

    # Loop through contrasts, outputting COPEs, statistic maps
    # and covariance maps.
    n_c = len(inputs['contrasts'])

    for i in range(0,n_c):

        # Read in contrast vector
        cvec = np.array(inputs['contrasts'][i]['c' + str(i+1)]['vector'])

        # Calculate C\hat{\beta}}
        cbeta = np.matmul(cvec, beta)
        print('contrast')
        print(cvec)
        print('beta')
        print(beta[:, 300000])
        print('cbeta')
        print(cbeta[300000])

        if inputs['contrasts'][i]['c' + str(i+1)]['statType'] == 'T':

            # A T contrast has only one row so we can output cbeta here
            cbeta = cbeta.reshape(
                        resms.shape[0],
                        resms.shape[1],
                        resms.shape[2],
                        )

            # Output cbeta/cope map
            cbetamap = nib.Nifti1Image(cbeta,
                                       nifti.affine,
                                       header=nifti.header)
            nib.save(cbetamap,
                os.path.join(OutDir, 
                    'blm_vox_beta_c' + str(i+1) + '.nii'))

            if not SVFlag:

                # Calculate c'(X'X)^(-1)c
                cvectiXtXcvec = np.matmul(
                    np.matmul(cvec, isumXtX),
                    np.transpose(cvec))

                # Calculate cov(c\hat{\beta})
                covcbeta = cvectiXtXcvec*resms

                # Output covariance map
                covcbetamap = nib.Nifti1Image(covcbeta,
                                              nifti.affine,
                                              header=nifti.header)
                nib.save(covcbetamap,
                    os.path.join(OutDir, 
                        'blm_vox_cov_c' + str(i+1) + '.nii'))

            else:

                # Calculate c'(X'X)^(-1)c
                cvectiXtXcvec = np.matmul(
                    np.matmul(cvec, isumXtX),
                    np.transpose(cvec))

                print('cvectiXtXcvec shape')
                print(cvectiXtXcvec.shape)

                # Calculate cov(c\hat{\beta})
                covcbeta = cvectiXtXcvec*resms.reshape(
                    resms.shape[0]*resms.shape[1]*resms.shape[2]
                    )

                covcbeta = covcbeta.reshape(
                    resms.shape[0],
                    resms.shape[1],
                    resms.shape[2]
                    )

                # Output covariance map
                covcbetamap = nib.Nifti1Image(covcbeta,
                                              nifti.affine,
                                              header=nifti.header)
                nib.save(covcbetamap,
                    os.path.join(OutDir, 
                        'blm_vox_cov_c' + str(i+1) + '.nii'))


            # To avoid division by zero errors we set the 
            # zero elements to one. XXXX ERRORS UNDER O LOOK INTO
            print(covcbeta[covcbeta<0])
            print(covcbeta.shape)
            print(np.where(covcbeta<0))
            tmp = covcbeta.reshape([n_v,1])
            print(np.where(tmp<0))
            print(sumXtX[np.where(tmp<0),:,:])
            print(isumXtX[np.where(tmp<0),:,:])
            covcbeta[covcbeta <= 0] = 1        

            # Calculate T statistic image

            tStatc = cbeta/np.sqrt(covcbeta)

            # Output statistic map
            tStatcmap = nib.Nifti1Image(tStatc,
                                        nifti.affine,
                                        header=nifti.header)
            nib.save(tStatcmap,
                os.path.join(OutDir, 
                    'blm_vox_Tstat_c' + str(i+1) + '.nii'))

        if inputs['contrasts'][i]['c' + str(i+1)]['statType'] == 'F':
        
            # Not spatially varying
            if not SVFlag:
                
                # Get dumension of cvector
                q = cvec.shape[0]

                # Calculate c'(X'X)^(-1)c
                # (Note C is read in the other way around for F)
                cvectiXtXcvec = np.matmul(
                    np.matmul(cvec, isumXtX),
                    np.transpose(cvec))

                # Cbeta needs to be nvox by 1 by npar for stacked
                # multiply.
                cbeta = cbeta.reshape(
                    cbeta.shape[0],
                    cbeta.shape[1],
                    1)
                cbeta = cbeta.transpose(1, 0, 2)

                # Calculate the inverse
                icvectiXtXcvec =blm_inverse(cvectiXtXcvec, ouflow=True)

                # Calculate the numerator of the F statistic
                Fnumerator = np.matmul(
                    cbeta.transpose(0, 2, 1),
                    np.matmul(icvectiXtXcvec, cbeta))
                # Fnumerator2 = np.matmul(
                #     cbeta.transpose(0, 2, 1),
                #     np.linalg.solve(cvectiXtXcvec, cbeta))
                Fnumerator = Fnumerator.reshape(Fnumerator.shape[0])

                # Calculate the denominator of the F statistic
                Fdenominator = (q*resms).reshape(
                    resms.shape[0]*resms.shape[1]*resms.shape[2])
                # Remove zeros in Fdenominator to avoid divide by 
                # zero errors
                Fdenominator[Fdenominator == 0] = 1

                # Calculate F statistic.
                fStatc = Fnumerator/Fdenominator
                fStatc = fStatc.reshape(
                    resms.shape[0],
                    resms.shape[1],
                    resms.shape[2]
                    )

                # Output statistic map
                fStatcmap = nib.Nifti1Image(fStatc,
                                            nifti.affine,
                                            header=nifti.header)
                nib.save(fStatcmap,
                    os.path.join(OutDir, 
                        'blm_vox_Fstat_c' + str(i+1) + '.nii'))




    # Clean up files
    os.remove(os.path.join(OutDir, 'nb.txt'))
    shutil.rmtree(os.path.join(OutDir, 'tmp'))

    w.resetwarnings()


# This function inverts matrix A. If ouflow is True,
# special handling is used to account for over/under
# flow.
def blm_inverse(A, ouflow=False):


    # If ouflow is true, we need to precondition A.
    if ouflow:

        # Calculate D, diagonal matrix with diagonal
        # elements D_ii equal to 1/sqrt(A_ii)
        D = np.zeros([3,3])
        np.fill_diagonal(D, 1/np.sqrt(A.diagonal()))

        # Precondition A.
        A = np.matmul(np.matmul(D, A), D)

    # np linalg inverse doesn't handle dim=[1,1]
    if np.ndim(A) == 1:
        iA = 1/A
    else:
        iA = np.linalg.inv(A)

    if ouflow:

        # Undo preconditioning.
        iA = np.matmul(np.matmul(D, iA), D)

    return(iA)

if __name__ == "__main__":
    main()
