#!/apps/well/R/3.4.3/bin/Rscript
library(MASS)
library(Matrix)
library(tictoc)

# ---------------------------------------------------------------------------------------
# IMPORTANT: Input options
# ---------------------------------------------------------------------------------------
#
# The below variables control which simulation is run and how. The variable names match
# those used in the `LMMPaperSim.py` file and are given as follows:
#
# - outDir: The output directory.
# - simInd: Simulation number.
# - batchNo: Batch number.
# 
# ---------------------------------------------------------------------------------------

# Read in arguments from commandline
args=(commandArgs(TRUE))

# Evaluate arguments
for(i in 1:length(args)){
  eval(parse(text=args[[i]]))
}

# Read in the fixed effects design
X <- read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/X.csv',sep=''),sep=',', header=FALSE)

# Get the number of columns of X/number of parameters in the model
n_p <- ncol(X)

# Read in the response vector
all_Y <- read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/Y_Rversion_',toString(batchNo),'.csv',sep=''),sep=',', header=FALSE)

# Number of voxels we have
nvox <- dim(all_Y)[2]

# Empty array for beta estimates
betas <- matrix(0,dim(all_Y)[2],n_p)

# Empty array for sigma2 estimates
sigma2 <- matrix(0,dim(all_Y)[2],1)

# Empty array for computation times
times <- matrix(0,dim(all_Y)[2],1)

# Empty array for T statistics
Tstats <- matrix(0,dim(all_Y)[2],1)

# Empty array for pvals
Pvals <- matrix(0,dim(all_Y)[2],1)

# Empty array for log-likelihoods
llh <- matrix(0,dim(all_Y)[2],1)

# Loop through each model and run lm for each voxel
for (i in 1:nvox){

  # Get Y
  y <- as.matrix(all_Y[,i])

  # If all y are zero this voxel was dropped from analysis as a
  # result of missing data
  if (!all(y==0)){

    # Get the indices of non-zero elements in y
    non_zero_indices <- which(y != 0)

    # Subset Y and X based on non-zero indices
    y <- y[non_zero_indices]
    X <- X[non_zero_indices, ]

    # Fit a linear regression model of y against 0 + X
    fit <- lm(y ~ 0 + ., data = as.data.frame(X))

    # Record fixed effects estimates
    betas[i, 1:n_p] <- coef(fit)

    # Record log likelihood
    llh[i, 1] <- logLik(fit)[1]

    # Create a contrast vector of length n_p consisting of all zeros, bar the last entry which is one
    contrast_vec <- rep(0, n_p)
    contrast_vec[n_p] <- 1

    # Compute the t-statistic and associated p-value for the contrast
    variance_of_contrast <- sum(contrast_vec %*% vcov(fit) %*% contrast_vec)
    Tstat <- sum(contrast_vec * coef(fit)) / sqrt(variance_of_contrast)
    df <- df.residual(fit)
    p <- 2 * pt(abs(Tstat), df = df, lower.tail = FALSE)
    sigma2[i, 1] <- summary(fit)$sigma^2

    # Make p-values 1 sided
    if (Tstat > 0) {
      p <- p/2
    } else {
      p <- 1-p/2
    }

    # Record p value
    Pvals[i, 1] <- p

    # Record T stat
    Tstats[i, 1] <- Tstat
    
  }

}

# Directory for lm results for this simulation
lmDir <- file.path(outDir, paste('sim',toString(simInd),sep=''),'lm')

# Make directory if it doesn't exist already
if (!file.exists(lmDir)) {
  dir.create(lmDir)
}

# Write results back to csv file
write.csv(betas,paste(lmDir,'/beta_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(sigma2,paste(lmDir,'/sigma2_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(llh,paste(lmDir,'/llh_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(Tstats,paste(lmDir,'/Tstat_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(Pvals,paste(lmDir,'/Pval_',toString(batchNo),'.csv',sep=''), row.names = FALSE)

# Remove the R file for this batch as we no longer need it
file.remove(paste(outDir,'/sim',toString(simInd),'/data/Y_Rversion_',toString(batchNo),'.csv',sep=''))