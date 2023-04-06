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

# Read in the response vector
all_Y <- read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/Y_Rversion_',toString(batchNo),'.csv',sep=''),sep=',', header=FALSE)

# Number of voxels we have
nvox <- dim(all_Y)[2]

# Empty array for beta estimates
betas <- matrix(0,dim(all_Y)[2],4)

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

# Loop through each model and run lmer for each voxel
for (i in 1:nvox){

  # Print i
  print(i)

  # Get Y
  y <- as.matrix(all_Y[,i])

  # If all y are zero this voxel was dropped from analysis as a
  # result of missing data
  if (!all(y==0)){

    # Reformat X into columns and mask
    x1 <- as.matrix(X[,1])[y!=0]
    x2 <- as.matrix(X[,2])[y!=0]
    x3 <- as.matrix(X[,3])[y!=0]
    x4 <- as.matrix(X[,4])[y!=0]

    # Finally, drop any missing Y
    y <- y[y!=0]

    # Fit the linear regression model
    fit <- lm(y ~ 0 + x1 + x2 + x3 + x4)

    # Record fixed effects estimates
    betas[i, 1:4] <- coef(fit)

    # Record log likelihood
    llh[i, 1] <- logLik(fit)[1]

    # Create a contrast vector
    contrast_vec <- c(0, 0, 0, 1)

    # Compute the t-statistic and associated p-value for the contrast
    Tstat <- sum(contrast_vec * coef(fit)) / sqrt(sum(contrast_vec^2 * vcovHC(fit)^2))
    df <- df.residual(fit)
    p <- 2 * pt(abs(Tstat), df = df, lower.tail = FALSE)

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