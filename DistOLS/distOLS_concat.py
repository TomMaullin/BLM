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
from DistOLS import distOLS_defaults

def main():

    # Change to distOLS directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    print(os.getcwd())

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
    inputs = distOLS_defaults.main()
    SVFlag = inputs[3]
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
    NIFTIsize = np.loadtxt(os.path.join("binputs","NIFTIsize.csv"), 
                        delimiter=",")

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

        # tmp code to output nifti
        nifti = nib.load(os.path.join("binputs", "example.nii"))

        betaimap = nib.Nifti1Image(betai,
                                   nifti.affine,
                                   header=nifti.header)
        nib.save(betaimap, 'beta' + str(i) + '.nii')

    del betai, betaimap, nifti

    if np.ndim(beta) == 0:
        beta = np.array([[beta]])
    elif np.ndim(beta) == 1:
        beta = np.array([beta])

    # Reshape beta along smallest axis for quicker
    # residual calculation
    beta_rs = np.zeros([beta.shape[1], beta.shape[0], 1])
    beta_rs_t = np.zeros([beta.shape[1], 1, beta.shape[0]])
    for i in range(0,beta.shape[0]):
        
       beta_rs[:, i, 0] = beta[i,:];
       beta_rs_t[:, 0, i] = beta[i,:];

    del beta

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

    # tmp code to output nifti
    nifti = nib.load(os.path.join("binputs", "example.nii"))

    ssmap = nib.Nifti1Image(ete.reshape(int(NIFTIsize[0]),
                                  int(NIFTIsize[1]),
                                  int(NIFTIsize[2])),
                            nifti.affine,
                            header=nifti.header)
    nib.save(ssmap, 'Residss.nii')
    
    w.resetwarnings()


if __name__ == "__main__":
    main()
