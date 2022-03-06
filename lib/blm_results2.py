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

def main3(*args):

    print('marker')

    t1 = time.time()
    print('started, time ',t1-t1)

    # Get job number
    jobNum = args[0]

    # Get practical number of voxel batches
    pnvb = args[1]

    # Work out number of batchs
    n_b = args[2]

    t1=time.time()

    # ----------------------------------------------------------------------
    # Check inputs
    # ----------------------------------------------------------------------
    if len(args)==3 or (not args[3]):
        # Load in inputs
        with open(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..',
                    'blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream,Loader=yaml.FullLoader)
    else:
        if type(args[3]) is str:
            # In this case inputs file is first argument
            with open(os.path.join(args[3]), 'r') as stream:
                inputs = yaml.load(stream,Loader=yaml.FullLoader)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[3]

    t2 = time.time()

    # ----------------------------------------------------------------------
    # Read basic inputs
    # ----------------------------------------------------------------------

    OutDir = inputs['outdir']

    with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
        print('inputs, time ',t2-t1, file=f)

    t1=time.time()

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

    # Get ns.
    X = loadFile(inputs['X'])
    n = X.shape[0]

    t2 = time.time()

    with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
        print('inputs2, time ',t2-t1, file=f)
        print('OutputCovB ', OutputCovB, file=f)


    # Load mask
    Mask = loadFile(os.path.join(OutDir,'blm_vox_mask.nii')).get_fdata()
    Mask = Mask.reshape(v, 1)

    # Load mask
    n_sv = loadFile(os.path.join(OutDir,'blm_vox_n.nii')).get_fdata()
    n_sv = n_sv.reshape(v, 1) # MARKER


    if 'analysis_mask' in inputs:

        amask_path = inputs["analysis_mask"]
        
        # Read in the mask nifti.
        amask = loadFile(amask_path).get_fdata().reshape([v,1])

    else:

        # By default make amask ones
        amask = np.ones([v,1])


    t1=time.time()

    # Get indices for whole analysis mask. These indices are the indices we
    # have recorded for the product matrices with respect to the entire volume
    amInds = get_amInds(amask)

    # ------------------------------------------------------------------------
    # Work out "Ring" and "Inner" indices for whole mask
    # ------------------------------------------------------------------------

    # Get indices of voxels in ring around brain where there are
    # missing studies.
    R_inds = np.sort(np.where((Mask==1)*(n_sv<n))[0])

    # Work out the 'ring' indices, in relation to the analysis mask
    ix_r = np.argsort(np.argsort(R_inds))
    R_inds_am = np.sort(np.where(np.in1d(amInds,R_inds))[0])[ix_r]

    # Get indices of the "inner" volume where all studies had information
    # present. I.e. the voxels (usually near the middle of the brain) where
    # every voxel has a reading for every study.
    I_inds = np.sort(np.where((Mask==1)*(n_sv==n))[0])

    # Work out the 'inner' indices, in relation to the analysis mask
    ix_i = np.argsort(np.argsort(I_inds))
    I_inds_am = np.sort(np.where(np.in1d(amInds,I_inds))[0])[ix_i]

    t2 = time.time()

    with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
        print('ring/inner, time ',t2-t1, file=f)

    # ------------------------------------------------------------------------
    # Number of voxels in ring and inner
    # ------------------------------------------------------------------------

    # Number of voxels in ring
    v_r = R_inds.shape[0]

    # Number of voxels in inner mask
    v_i = I_inds.shape[0]

    # Number of voxels in whole (inner + ring) mask
    v_m = v_i + v_r

    # ------------------------------------------------------------------------
    # Degrees of freedom (n-p)
    # ------------------------------------------------------------------------

    # Get df map
    df_r = n_sv[R_inds,:] - p
    df_r = df_r.reshape([v_r])
    df_i = n - p

    # ------------------------------------------------------------------------
    # The next operations are more computationally intensive so we split 
    # computation into blocks of voxels
    # ------------------------------------------------------------------------


    t1=time.time()

    # ------------------------------------------------------------------------
    # Work out block of voxels we are looking at
    # ------------------------------------------------------------------------
    # Get indices for block. These indices have to be the indices we want to
    # compute, in relation to the entire volume. If we aren't partitioning by 
    # block these will be equal to amInds
    bamInds = get_amInds(amask, jobNum, pnvb) 

    # ------------------------------------------------------------------------
    # Number of contrasts
    # ------------------------------------------------------------------------
    c = len(inputs['contrasts'])

    # Record how many T contrasts and F contrasts we have seen
    nt = 0
    nf = 0
    for i in range(0,c):

        # Read in contrast vector
        Lvec = str2vec(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        Lvec = np.array(Lvec)

        if Lvec.ndim == 1:
            nt = nt + 1
        else:
            nf = nf + 1


    t2 = time.time()


    with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
        print('contrasts, time ',t2-t1, file=f)

        print('c ', c, file=f)
        print('nt ', nt, file=f)
        print('nf ', nf, file=f)

    # ------------------------------------------------------------------------
    # Output volume dimensions
    # ------------------------------------------------------------------------

    # Dimension of beta volume
    dimBeta = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],p)

    if OutputCovB:

        # Dimension of cov(beta) NIFTI
        dimCov = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],p**2)

    # Work out the dimension of the T-stat-related volumes
    dimT = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],nt)

    # Work out the dimension of the F-stat-related volumes
    dimF = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],nf)

    # ------------------------------------------------------------------------
    # Split the voxels into computable groups
    # ------------------------------------------------------------------------

    t1=time.time()

    # Work out the number of voxels we can actually compute at a time.
    # (This is really just a rule of thumb guess but works reasonably in
    # practice).
    nvb = MAXMEM/(10*8*(p**2))
    
    # Work out number of groups we have to split indices into.
    nvg = int(len(bamInds)//nvb+1)


    with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
        print('nvg: ', nvg, file=f)

    # Split voxels we want to look at into groups we can compute
    voxelGroups = np.array_split(bamInds, nvg)

    # Loop through list of voxel indices, looking at each group of voxels, in
    # turn.
    for cv in range(nvg):

        t1 = time.time()


        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('in loop ', file=f)

            print('cv ', cv, file=f)

        # Current group of voxels
        bamInds_cv = voxelGroups[cv]

        # Mask for current voxels
        Mask_cv = np.array(Mask)
        Mask_cv[~np.in1d(np.arange(v).reshape(v,1), bamInds_cv)]=0

        # Get indices of voxels in ring around brain where there are
        # missing studies.
        R_inds = np.sort(np.where((Mask_cv==1)*(n_sv<n))[0])

        # Work out the 'ring' indices, in relation to the analysis mask
        ix_r = np.argsort(np.argsort(R_inds))
        R_inds_am = np.sort(np.where(np.in1d(amInds,R_inds))[0])[ix_r]

        # Get indices of the "inner" volume where all studies had information
        # present. I.e. the voxels (usually near the middle of the brain) where
        # every voxel has a reading for every study.
        I_inds = np.sort(np.where((Mask_cv==1)*(n_sv==n))[0])

        # Work out the 'inner' indices, in relation to the analysis mask
        ix_i = np.argsort(np.argsort(I_inds))
        I_inds_am = np.sort(np.where(np.in1d(amInds,I_inds))[0])[ix_i]

        t2 = time.time()


        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('in loop masked inds, time ',t2-t1, file=f)

        t1 = time.time()

        # ------------------------------------------------------------------------
        # Number of voxels in ring and inner
        # ------------------------------------------------------------------------

        # Number of voxels in ring
        v_r = R_inds.shape[0]

        # Number of voxels in inner mask
        v_i = I_inds.shape[0]

        # Number of voxels in whole (inner + ring) mask
        v_m = v_i + v_r

        # --------------------------------------------------------------------------------
        # Load X'X, X'Y, Y'Y
        # --------------------------------------------------------------------------------

        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('in loop got vox numbers, time ',t2-t1, file=f)

        t1 = time.time()

        # Ring X'Y, Y'Y
        XtY_r = readLinesFromNPY(os.path.join(OutDir,"tmp",'XtY.npy'), R_inds_am).reshape([v_r, p, 1])

        t2 = time.time()

        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('in loop got XtY_r, time ',t2-t1, file=f)

        t1 = time.time()

        YtY_r = readLinesFromNPY(os.path.join(OutDir,"tmp",'YtY.npy'), R_inds_am).reshape([v_r, 1, 1])

        t2 = time.time()

        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('in loop got YtY_r, time ',t2-t1, file=f)

        t1 = time.time()

        # Inner X'Y, Y'Y
        XtY_i = readLinesFromNPY(os.path.join(OutDir,"tmp",'XtY.npy'), I_inds_am).reshape([v_i, p, 1])
        t2 = time.time()

        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('in loop got XtY_i, time ',t2-t1, file=f)

        t1 = time.time()

        YtY_i = readLinesFromNPY(os.path.join(OutDir,"tmp",'YtY.npy'), I_inds_am).reshape([v_i, 1, 1])

        t2 = time.time()

        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('in loop got YtY_i, time ',t2-t1, file=f)

            print('v_i ', v_i, file=f)
            print('v_r ', v_r, file=f)

        if v_r:

            t1 = time.time()
            # Ring X'X
            XtX_r = readAndSumUniqueAtB('XtX', OutDir, R_inds, n_b, True).reshape([v_r, p, p])

            t2 = time.time()

            with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
                print('in loop got XtX_r time ',t2-t1, file=f)

            # ----------------------------------------------------------------------------
            # Remove low rank designs
            # ----------------------------------------------------------------------------

            t1 = time.time()
            # Work out indices of low rank designs
            lowrank_inds = np.where(np.linalg.matrix_rank(XtX_r)<p)[0]
            fullrank_inds = np.where(np.linalg.matrix_rank(XtX_r)==p)[0]

            # Work out number of low rank indices
            v_lowrank = np.prod(lowrank_inds.shape)

            t2 = time.time()

            with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
                print('in loop removed rank time ',t2-t1, file=f)

                print('v_lowrank ', v_lowrank, file=f)

            # If we have low rank indices remove them from our working variables
            if v_lowrank:

                t1 = time.time()

                # Remove low rank designs from the existing NIFTI files
                addBlockToNifti(os.path.join(OutDir, 'blm_vox_mask.nii'), np.zeros(v_lowrank), R_inds[lowrank_inds],volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)
                addBlockToNifti(os.path.join(OutDir, 'blm_vox_edf.nii'), np.zeros(v_lowrank), R_inds[lowrank_inds],volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)
                addBlockToNifti(os.path.join(OutDir, 'blm_vox_n.nii'), np.zeros(v_lowrank), R_inds[lowrank_inds],volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)
            
                # Remove from R_inds
                R_inds = R_inds[fullrank_inds]
            
                # Remove from product matrices
                YtY_r = YtY_r[fullrank_inds,:,:]
                XtX_r = XtX_r[fullrank_inds,:,:]
                XtY_r = XtY_r[fullrank_inds,:,:]
                
                # Recalculate number of voxels left in ring
                v_r = R_inds.shape[0]
        

                t2 = time.time()

                with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
                    print('in loop updated masks ',t2-t1, file=f)


        if v_i:

            t1 = time.time()

            # Inner X'X
            XtX_i = readAndSumUniqueAtB('XtX', OutDir, I_inds, n_b, False).reshape([1, p, p])

            # Check the design is full rank
            if np.linalg.matrix_rank(XtX_i)<p:
                raise Exception('The design matrix, X, is of insufficient rank.') 


            t2 = time.time()

            with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
                print('in loop inner rank check ',t2-t1, file=f)

        # ----------------------------------------------------------------------
        # Calculate betahat = (X'X)^(-1)X'Y and output beta maps
        # ----------------------------------------------------------------------    

        # Get beta for ring
        if v_r:

            t1 = time.time()

            # Calculate masked Beta for ring
            beta_r = np.linalg.solve(XtX_r, XtY_r)

            # Output Beta
            addBlockToNifti(os.path.join(OutDir, 'blm_vox_beta.nii'), beta_r, R_inds,volInd=None,dim=dimBeta,aff=nifti.affine,hdr=nifti.header)

            t2 = time.time()

            with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
                print('in loop beta ring ',t2-t1, file=f)


        # If we have indices where all studies are present, work out X'X and
        # X'Y for these studies.
        if v_i:

            t1 = time.time()

            # Calculate masked Beta for ring
            beta_i = np.linalg.solve(XtX_i, XtY_i)


            # Output Beta
            addBlockToNifti(os.path.join(OutDir, 'blm_vox_beta.nii'), beta_i, I_inds,volInd=None,dim=dimBeta,aff=nifti.affine,hdr=nifti.header)

            t2 = time.time()

            with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
                print('in loop beta inner ',t2-t1, file=f)

        # ----------------------------------------------------------------------
        # Calculate residual sum of squares e'e = Y'Y - (Xb)'Xb
        # ---------------------------------------------------------------------- 

        if v_i:

            # Reshape beta along smallest axis for quicker
            # residual calculation
            beta_i_t = beta_i.transpose(0,2,1)

            # Calculate Beta transpose times XtX and delete the
            # now redundant matrices.
            betatXtX_i = np.matmul(beta_i_t, XtX_i)
            del beta_i_t

            # Multiply BetatXtX by Beta and delete the redundant
            # matrices.
            betatXtXbeta_i = np.matmul(betatXtX_i, beta_i)
            del betatXtX_i

            # Reshape betat XtX beta
            betatXtXbeta_i = np.reshape(betatXtXbeta_i, [v_i,1])

            # Residual sum of squares
            ete_i = YtY_i.reshape([v_i,1]) - betatXtXbeta_i
            del betatXtXbeta_i

        if v_r:

            # Reshape beta along smallest axis for quicker
            # residual calculation
            beta_r_t = beta_r.transpose(0,2,1)

            # Calculate Beta transpose times XtX and delete the
            # now redundant matrices.
            betatXtX_r = np.matmul(beta_r_t, XtX_r)
            del beta_r_t

            # Multiply BetatXtX by Beta and delete the redundant
            # matrices.
            betatXtXbeta_r = np.matmul(betatXtX_r, beta_r)
            del betatXtX_r

            # Reshape betat XtX beta
            betatXtXbeta_r = np.reshape(betatXtXbeta_r, [v_r,1])

            # Residual sum of squares
            ete_r = YtY_r.reshape([v_r,1]) - betatXtXbeta_r
            del betatXtXbeta_r

        # ----------------------------------------------------------------------
        # Calculate residual mean squares = e'e/(n - p)
        # ----------------------------------------------------------------------

        # Mask spatially varying n
        if v_r:
            n_sv_r = n_sv[R_inds,:]

            # In spatially varying the degrees of freedom
            # varies across voxels
            resms_r = ete_r/(n_sv_r-p)
            addBlockToNifti(os.path.join(OutDir, 'blm_vox_resms.nii'), resms_r, R_inds,volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

        if v_i:

            # All voxels in the inner mask have n scans present
            resms_i = ete_i/(n-p)
            addBlockToNifti(os.path.join(OutDir, 'blm_vox_resms.nii'), resms_i, I_inds,volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)


        # ----------------------------------------------------------------------
        # Calculate log likelihood
        # ----------------------------------------------------------------------

        if v_r:

            sigma2_r = 1/n_sv_r * ete_r.reshape(n_sv_r.shape)

            # Work out -1/2(nln(sigma^2))
            firstterm = -0.5*(n_sv_r.reshape(sigma2_r.shape)*np.log(sigma2_r)).reshape(ete_r.shape)

            # Work out -N/2 ln(2pi)
            secondterm = -0.5*(n_sv_r.reshape(sigma2_r.shape)*np.log(2*np.pi)).reshape(ete_r.shape)  

            # Work out the log likelihood
            llh_r = firstterm + secondterm - 0.5*n_sv_r.reshape(ete_r.shape)

            # Output
            addBlockToNifti(os.path.join(OutDir, 'blm_vox_llh.nii'), llh_r, R_inds,volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

        if v_i:

            sigma2_i = 1/n * ete_i
        
            # Work out -1/2(nln(sigma^2))
            firstterm = -0.5*(n*np.log(sigma2_i)).reshape(ete_i.shape)

            # Work out -N/2 ln(2pi)
            secondterm = -0.5*(n*np.log(2*np.pi))

            # Work out the log likelihood
            llh_i = firstterm + secondterm - 0.5*n

            # Output
            addBlockToNifti(os.path.join(OutDir, 'blm_vox_llh.nii'), llh_i, I_inds,volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

        # ----------------------------------------------------------------------
        # Calculate beta covariance maps
        # ----------------------------------------------------------------------
            
        t1 = time.time()

        # Calculate masked (x'X)^(-1) values for ring
        if v_r:
            iXtX_r = blm_inverse(XtX_r, ouflow=True)
        if v_i:
            iXtX_i = blm_inverse(XtX_i, ouflow=True)

        t2 = time.time()

        with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
            print('inverse ',t2-t1, file=f)

        if OutputCovB:
            
            vol = 0

            # Output variance for each pair of betas
            for i in range(0,p):
                for j in range(0,p):

                        # Unmask cov beta ij
                        covbetaij = np.zeros([v])

                        if v_r: 
                            # Calculate masked cov beta ij for ring
                            covbetaij_r = np.multiply(
                                resms_r.reshape([resms_r.shape[0]]),
                                iXtX_r[:,i,j])

                            # Output 
                            addBlockToNifti(os.path.join(OutDir, 'blm_vox_cov.nii'), covbetaij_r, R_inds,volInd=vol,dim=dimCov,aff=nifti.affine,hdr=nifti.header)
            
                        if v_i:
                            # Calculate masked cov beta ij for inner
                            covbetaij_i = np.multiply(
                                resms_i.reshape([resms_i.shape[0]]),
                                iXtX_i[:,i,j])

                            # Output 
                            addBlockToNifti(os.path.join(OutDir, 'blm_vox_cov.nii'), covbetaij_i, I_inds,volInd=vol,dim=dimCov,aff=nifti.affine,hdr=nifti.header)

                        vol = vol+1;

        # ----------------------------------------------------------------------
        # Calculate COPEs, statistic maps and covariance maps.
        # ----------------------------------------------------------------------

        # Current number for contrast (T and F)
        current_nt = 0
        current_nf = 0

        for i in range(0,c):

            # Read in contrast vector
            # Get number of parameters
            Lvec = str2vec(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
            Lvec = np.array(Lvec)

            # Calculate L\hat{\beta}}
            if v_r:
                Lbeta_r = np.matmul(Lvec, beta_r)
            if v_i:
                Lbeta_i = np.matmul(Lvec, beta_i)
        
            if Lvec.ndim == 1:
                statType='T'
                Lvec = Lvec.reshape([1,Lvec.shape[0]])
            else:
                statType='F'

            if statType == 'T':

                if v_r:

                    # A T contrast has only one row so we can output Lbeta here
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_con.nii'), Lbeta_r, R_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                    # Calculate c'(X'X)^(-1)c
                    LvectiXtXLvec_r = np.matmul(
                        np.matmul(Lvec, iXtX_r),
                        np.transpose(Lvec)).reshape(v_r)

                    # Calculate masked cov(c\hat{\beta}) for ring
                    covLbeta_r = LvectiXtXLvec_r*resms_r.reshape(v_r)
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conSE.nii'), np.sqrt(covLbeta_r), R_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                    # Calculate T stat
                    tStatc_r = Lbeta_r.reshape(v_r)/np.sqrt(covLbeta_r)
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conT.nii'), tStatc_r, R_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                    # Degrees of freedom
                    df_r = n_sv[R_inds,:] - p
                    df_r = df_r.reshape(df_r.shape[0])

                    # Calculate p (seperately for >0 and <0 to avoid underflow)
                    pc_r = np.zeros(np.shape(tStatc_r))
                    pc_r[tStatc_r < 0] = -np.log10(1-stats.t.cdf(tStatc_r[tStatc_r < 0], df_r[tStatc_r < 0]))
                    pc_r[tStatc_r >= 0] = -np.log10(stats.t.cdf(-tStatc_r[tStatc_r >= 0], df_r[tStatc_r >= 0]))

                    # Remove infs
                    if "minlog" in inputs:
                        pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=inputs['minlog']
                    else:
                        pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=-323.3062153431158

                    # Output p
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conTlp.nii'), pc_r, R_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                if v_i:

                    # A T contrast has only one row so we can output Lbeta here
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_con.nii'), Lbeta_i, I_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)
                    
                    # Calculate c'(X'X)^(-1)c
                    LvectiXtXLvec_i = np.matmul(
                        np.matmul(Lvec, iXtX_i),
                        np.transpose(Lvec))

                    # Calculate masked cov(c\hat{\beta}) for inner
                    covLbeta_i = LvectiXtXLvec_i*resms_i.reshape(v_i)
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conSE.nii'), np.sqrt(covLbeta_i), I_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                    # Calculate T stat
                    tStatc_i = Lbeta_i.reshape(v_i)/np.sqrt(covLbeta_i)
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conT.nii'), tStatc_i, I_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                    # Calculate p (seperately for >0 and <0 to avoid underflow)
                    pc_i = np.zeros(np.shape(tStatc_i))
                    pc_i[tStatc_i < 0] = -np.log10(1-stats.t.cdf(tStatc_i[tStatc_i < 0], df_i))
                    pc_i[tStatc_i >= 0] = -np.log10(stats.t.cdf(-tStatc_i[tStatc_i >= 0], df_i))

                    # Remove infs
                    if "minlog" in inputs:
                        pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=inputs['minlog']
                    else:
                        pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=-323.3062153431158

                    # Output p
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conTlp.nii'), pc_i, I_inds,volInd=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                # Record that we have seen another T contrast
                current_nt = current_nt + 1

            if statType == 'F':

                # Get dimension of Lvector
                q = Lvec.shape[0]
            
                # Calculate L'(X'X)^(-1)L
                # (Note L is read in the other way around for F)
                if v_r:

                    LvectiXtXLvec_r = np.matmul(
                        np.matmul(Lvec, iXtX_r),
                        np.transpose(Lvec))

                    # Lbeta needs to be nvox by 1 by npar for stacked
                    # multiply.
                    Lbetat_r = Lbeta_r.transpose(0,2,1)

                    # Calculate masked (L'(X'X)^(-1)L)^(-1) values for ring
                    iLvectiXtXLvec_r = blm_inverse(LvectiXtXLvec_r, ouflow=True).reshape([v_r, q*q])

                    # Calculate the numerator of the F statistic for the ring
                    Fnumerator_r = np.matmul(Lbetat_r, np.linalg.solve(LvectiXtXLvec_r, Lbeta_r))
                    Fnumerator_r = Fnumerator_r.reshape(Fnumerator_r.shape[0])

                    # Calculate the denominator of the F statistic for ring
                    Fdenominator_r = q*resms_r.reshape([v_r])

                    # Calculate F statistic.
                    fStatc_r = Fnumerator_r/Fdenominator_r

                    # Output F statistic.
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conF.nii'), fStatc_r, R_inds,volInd=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                    # Work out p for this contrast
                    pc_r = -np.log10(1-stats.f.cdf(fStatc_r, q, df_r))

                    # Remove infs
                    if "minlog" in inputs:
                        pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=inputs['minlog']
                    else:
                        pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=-323.3062153431158

                    # Output p
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conFlp.nii'), pc_r, R_inds,volInd=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                    # Reshape for partial R^2
                    n_sv_r = n_sv_r.reshape([v_r])

                    # Calculate partial R2 masked for ring.
                    partialR2_r = (q*fStatc_r)/(q*fStatc_r + n_sv_r - p)

                    # Output R^2
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conR2.nii'), partialR2_r, R_inds,volInd=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)


                if v_i:

                    LvectiXtXLvec_i = np.matmul(
                        np.matmul(Lvec, iXtX_i),
                        np.transpose(Lvec))

                    # Lbeta needs to be nvox by 1 by npar for stacked
                    # multiply.
                    Lbetat_i = Lbeta_i.transpose(0,2,1)

                    # Calculate masked (L'(X'X)^(-1)L)^(-1) values for inner
                    iLvectiXtXLvec_i = blm_inverse(LvectiXtXLvec_i, ouflow=True).reshape([1, q*q])

                    # Calculate the numerator of the F statistic for the ring
                    Fnumerator_i = np.matmul(Lbetat_i, np.linalg.solve(LvectiXtXLvec_i, Lbeta_i))
                    Fnumerator_i = Fnumerator_i.reshape(Fnumerator_i.shape[0])

                    # Calculate the denominator of the F statistic for inner
                    Fdenominator_i = q*resms_i.reshape([v_i])

                    # Calculate F statistic.
                    fStatc_i = Fnumerator_i/Fdenominator_i

                    # Output F statistic.
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conF.nii'), fStatc_i, I_inds,volInd=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                    # Degrees of freedom
                    df_r = n_sv[R_inds,:] - p
                    df_r = df_r.reshape(df_r.shape[0])

                    # Work out p for this contrast
                    pc_i = -np.log10(1-stats.f.cdf(fStatc_i, q, df_i))

                    # Remove infs
                    if "minlog" in inputs:
                        pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=inputs['minlog']
                    else:
                        pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=-323.3062153431158

                    # Output p
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conFlp.nii'), pc_i, I_inds,volInd=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                    # Calculate partial R2 masked for inner mask.
                    partialR2_i = (q*fStatc_i)/(q*fStatc_i + n - p)

                    # Output R^2
                    addBlockToNifti(os.path.join(OutDir, 'blm_vox_conR2.nii'), partialR2_i, I_inds,volInd=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

    w.resetwarnings()


# This function inverts matrix A. If ouflow is True,
# special handling is used to account for over/under
# flow. In this case, it assumes that A has non-zero
# diagonals.
def blm_inverse(A, ouflow=False):

    # Work out number of matrices and dimension of
    # matrices. I.e. if we have seven 3 by 3 matrices
    # to invert n_r = 7, d_r = 3.
    n_r = A.shape[0]
    d_r = A.shape[1]

    # If ouflow is true, we need to precondition A.
    if ouflow:

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


# ============================================================================
#
# For a specified set of voxels, the below function reads in the unique 
# product matrices A'B from each batch job, works out which voxel had which 
# product matrix, sums the batch product matrices and returns the sum, i.e. 
# the product matrix for the entire analysis, at each voxel.
#
# Note: This function is only really designed for the product matrix X'X in
# BLM. This function originates from BLMM, where it is also used for other
# matrices.
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
# - `vinds`: Voxel indices; (flattened) indices representing which voxels we 
#            are interested in looking at.
# - `n_b`: The number of batches run during the batch stage.
# - `sv`: Spatial varying boolean value. This tells us if we expect the
#         product matrix to vary across these voxels, or whether we expect it
#         to be the same for all of them.
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
def readAndSumUniqueAtB(AtBstr, OutDir, vinds, n_b, sv):

    # Work out the uniqueness mask for the spatially varying designs
    uniquenessMask = loadFile(os.path.join(OutDir,"tmp", 
        "blm_vox_uniqueM_batch1.nii")).get_fdata()

    v = np.prod(uniquenessMask.shape)
    vcurrent = np.prod(vinds.shape)

    uniquenessMask=uniquenessMask.reshape(v)

    # Work out how many unique matrices there were
    maxM = np.int32(np.amax(uniquenessMask))

    if sv:
        # Work out the uniqueness mask inside the ring around the brain
        uniquenessMask = uniquenessMask[vinds]
    else:
        # Work out the uniqueness mask value inside the inner part of the brain
        uniquenessMask = uniquenessMask[vinds[0]] 


    # read in XtX
    AtB_batch_unique = np.load(
        os.path.join(OutDir,"tmp",AtBstr+"1.npy"))

    # Make zeros for outer ring of brain XtX (remember A'B is still flattened)
    if sv:
        AtB = np.zeros((vcurrent, AtB_batch_unique.shape[1]))

    # Fill with unique maskings
    for m in range(1,maxM+1):

        if sv:
            # Work out X'X for the ring
            AtB[np.where(uniquenessMask==m),:] = AtB_batch_unique[(m-1),:]

        # Work out X'X for the inner
        else:
            if uniquenessMask == m:
                AtB = AtB_batch_unique[(m-1),:]

    # Cycle through batches and add together results.
    for batchNo in range(2,(n_b+1)):

        # Read in uniqueness Mask file
        uniquenessMask = loadFile(os.path.join(OutDir,"tmp", 
            "blm_vox_uniqueM_batch" + str(batchNo) + ".nii")).get_fdata().reshape(v)

        maxM = np.int32(np.amax(uniquenessMask))

        if sv:
            # Work out the uniqueness mask inside the ring around the brain
            uniquenessMask = uniquenessMask[vinds] 
        else:
            # Work out the uniqueness mask value inside the inner part of the brain
            uniquenessMask = uniquenessMask[vinds[0]] 


        # read in XtX
        AtB_batch_unique = np.load(
            os.path.join(OutDir,"tmp",AtBstr + str(batchNo) + ".npy"))

        # Make zeros for whole nifti XtX
        if sv:
            AtB_batch = np.zeros((vcurrent, AtB_batch_unique.shape[1]))

        # Fill with unique maskings
        for m in range(1,maxM+1):

            with open(os.path.join(OutDir,'results' + str(jobNum) + '.txt'), 'a') as f:
                print('maxM ',maxM, file=f)

            if sv:
                AtB_batch[np.where(uniquenessMask==m),:] = AtB_batch_unique[(m-1),:]
            else:
                # Work out X'X for the inner
                if uniquenessMask == m:

                    AtB_batch = AtB_batch_unique[(m-1),:]

        # Add to running total
        AtB = AtB + AtB_batch

    return(AtB)

if __name__ == "__rain__":
    main()