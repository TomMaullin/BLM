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

def main():

    # Change to blm directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    # Read the matrices from the first batch.
    sumXtX = np.loadtxt(os.path.join("binputs","XtX1.csv"), 
                        delimiter=",")
    sumXtY = np.loadtxt(os.path.join("binputs","XtY1.csv"), 
                        delimiter=",")
    sumYtY = np.loadtxt(os.path.join("binputs","YtY1.csv"), 
                        delimiter=",")

    # Delete the files as they are no longer needed.
    os.remove(os.path.join("binputs","XtX1.csv"))
    os.remove(os.path.join("binputs","XtY1.csv"))
    os.remove(os.path.join("binputs","YtY1.csv"))

    # Work out how many files we need.
    XtX_files = glob.glob(os.path.join("binputs","XtX*"))

    # Cycle through batches and add together results.
    for batchNo in range(2,(len(XtX_files)+2)):
        
        # Sum the batches.
        sumXtX = sumXtX + np.loadtxt(
            os.path.join("binputs","XtX" + str(batchNo) + ".csv"), 
                         delimiter=",")

        sumXtY = sumXtY + np.loadtxt(
            os.path.join("binputs","XtY" + str(batchNo) + ".csv"), 
                         delimiter=",")

        sumYtY = sumYtY + np.loadtxt(
            os.path.join("binputs","YtY" + str(batchNo) + ".csv"), 
                         delimiter=",")
        
        # Delete the files as they are no longer needed.
        os.remove(os.path.join(os.getcwd(), "binputs","XtX" + str(batchNo) + ".csv"))
        os.remove(os.path.join(os.getcwd(), "binputs","XtY" + str(batchNo) + ".csv"))
        os.remove(os.path.join(os.getcwd(), "binputs","YtY" + str(batchNo) + ".csv"))

    # Dimension bug handling
    if np.ndim(sumXtX) == 0:
        sumXtX = np.array([[sumXtX]])
    elif np.ndim(sumXtX) == 1:
        sumXtX = np.array([sumXtX])

    if np.ndim(sumXtY) == 0:
        sumXtY = np.array([[sumXtY]])
    elif np.ndim(sumXtY) == 1:
        sumXtY = np.array([sumXtY])

    # Mask and reshape if we are using a spatially varying design.
    with open('blm_defaults.yml', 'r') as stream:
        inputs = yaml.load(stream)
    SVFlag = inputs['SVFlag']
    if SVFlag:

        # Remove zero lines and convert back to number voxels (in
        # mask) by number of parametes by number of parameters)
        sumXtX = sumXtX.reshape([sumXtX.shape[0], 
                     int(np.sqrt(sumXtX.shape[1])),
                     int(np.sqrt(sumXtX.shape[1]))])
        sumXtX_m = sumXtX[np.where(np.linalg.det(sumXtX)!=0)[0]]
        
        isumXtX_m = np.linalg.inv(sumXtX_m).reshape(
                      [sumXtX_m.shape[0],
                       int(sumXtX_m.shape[1])*int(sumXtX_m.shape[2])])

        isumXtX = np.zeros([sumXtX.shape[0],
                            int(sumXtX.shape[1])*int(sumXtX.shape[2])])

        isumXtX[np.where(np.linalg.det(sumXtX)!=0)[0]]=isumXtX_m

        isumXtX = isumXtX.reshape([isumXtX.shape[0],
                                   int(np.sqrt(isumXtX.shape[1])),
                                   int(np.sqrt(isumXtX.shape[1]))])


    # If we are not using a spatially varying design, inverse in
    # the normal manner.
    else:
        # np linalg inverse doesn't handle dim=[1,1]
        if np.ndim(sumXtX) == 1:
            isumXtX = 1/sumXtX
        else:
            isumXtX = np.linalg.inv(sumXtX)

    # Read in the nifti size.
    with open(inputs['Y_files']) as a:
        nifti = nib.load(a.readline().replace('\n', ''))

    NIFTIsize = nifti.shape

    # If we are doing spatially varying we need to reshape XtY.
    if SVFlag:
        sumXtY = sumXtY.transpose()
        sumXtY = sumXtY.reshape([sumXtY.shape[0], sumXtY.shape[1], 1])
    
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
        nib.save(betaimap, 'blm_vox_beta_b' + str(i+1) + '.nii')

    del betai, betaimap

    if np.ndim(beta) == 0:
        beta = np.array([[beta]])
    elif np.ndim(beta) == 1:
        beta = np.array([beta])

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
    del beta_rs_t, sumXtX

    # Multiply BetatXtX by Beta and delete the reduundant
    # matrices.
    betatXtXbeta = np.matmul(betatXtX, beta_rs)
    del betatXtX, beta_rs

    # Reshape betat XtX beta
    betatXtXbeta = np.reshape(betatXtXbeta, betatXtXbeta.shape[0])

    # Residual sum of squares
    ete = sumYtY - betatXtXbeta
    ete = ete.reshape(int(NIFTIsize[0]),
                      int(NIFTIsize[1]),
                      int(NIFTIsize[2]))

    print(ete.shape)

    # Get residual mean squares by dividing by degrees of
    # freedom
    if not SVFlag:

        # Get number of scans and number of parameters
        X = np.loadtxt(inputs['X'], delimiter=',')
        n_s = X.shape[0]
        n_p = X.shape[1]

        # In non spatially varying the degrees of freedom
        # are fixed across voxels
        resms = ete/(n_s-n_p)

    else:
        
        # Get number of scans and number of parameters
        X = np.loadtxt(inputs['X'], delimiter=',')
        n_s = X.shape[0]
        n_p = X.shape[1]

        # Load in the spatially varying number of scans.
        n_s = nib.load('blm_vox_nsv.nii')
        n_s = n_s.get_data()

        print(repr(n_s))

        # To avoid division by zero errors we set the 
        # zero elements to one.
        n_s[n_s == 0] = 1

        print(repr(n_s))

        # In spatially varying the degrees of freedom
        # varies across voxels
        resms = ete/(n_s-n_p)

        print(repr(resms))

    # Output ResSS.
    msmap = nib.Nifti1Image(resms,
                            nifti.affine,
                            header=nifti.header)
    nib.save(msmap, 'blm_vox_resms.nii')

    # calculate beta covariance maps
    print(isumXtX.shape)
    print(resms.shape)
    print(isumXtX)
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
                        'blm_vox_cov_b' + str(i+1) + ',' + str(j+1) + '.nii')

        del covbetaijmap

    else:

        # Output variance for each pair of betas
        for i in range(0,isumXtX.shape[1]):
            for j in range(0,isumXtX.shape[2]):

                    print(repr(isumXtX[:,i,j]))
                    print(repr(isumXtX[:,i,j].shape))

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
                        'blm_vox_cov_b' + str(i+1) + ',' + str(j+1) + '.nii')

        del covbetaijmap

        print('tmp')

    print(inputs['contrasts'])
    print(inputs['contrasts'][0])
    print(inputs['contrasts'][0]['c1']['vector'])

    # Loop through contrasts, outputting COPEs, statistic maps
    # and covariance maps.
    n_c = len(inputs['contrasts'])
    print(n_c)

    for i in range(0,n_c):

        # Read in contrast vector
        cvec = np.array(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        print(cvec)
        print(type(cvec))
        #cvec = eval(cvec.replace(' ',''))

        print(cvec)
        print(type(cvec))

        # Calculate C\hat{\beta}}
        cbeta = np.matmul(cvec, beta)
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
            'blm_vox_beta_c' + str(i+1) + '.nii')


        if not SVFlag:

            # Calculate c'(X'X)^(-1)c
            cvectiXtXcvec = np.matmul(
                np.matmul(np.transpose(cvec), isumXtX),
                cvec)

            print(cvectiXtXcvec)
            print(beta.shape)
            print(repr(beta))

            # Calculate cov(c\hat{\beta})
            covbetac = cvectiXtXcvec*resms
            print(covbetac.shape)

            # Output covariance map
            covbetacmap = nib.Nifti1Image(covbetac,
                                          nifti.affine,
                                          header=nifti.header)
            nib.save(covbetacmap,
                'blm_vox_cov_c' + str(i+1) + '.nii')

        else:

            print(isumXtX.shape)

            # Calculate c'(X'X)^(-1)c
            cvectiXtXcvec = np.matmul(
                np.matmul(np.transpose(cvec), isumXtX),
                cvec)

            print(cvectiXtXcvec)
            print(beta.shape)
            print(repr(beta))

            # Calculate cov(c\hat{\beta})
            covbetac = cvectiXtXcvec*resms.reshape(
                resms.shape[0]*resms.shape[1]*resms.shape[2]
                )
            print(covbetac.shape)

            # Output covariance map
            covbetacmap = nib.Nifti1Image(covbetac,
                                          nifti.affine,
                                          header=nifti.header)
            nib.save(covbetacmap,
                'blm_vox_cov_c' + str(i+1) + '.nii')        


    w.resetwarnings()


if __name__ == "__main__":
    main()
