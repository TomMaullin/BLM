# This function takes in the X matrix for a given block and 
# returns X transpose X.
# ==============================================================
# USAGE: XtX = blkXtX(X)
# --------------------------------------------------------------
# It takes the following inputs:
# 	X - the design matrix.
# ==============================================================
# Author: Tom Maullin
import numpy as np

def blkXtX(X):

    return np.transpose(X) @ X