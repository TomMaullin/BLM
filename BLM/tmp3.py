import glob
import numpy as np
import nibabel as nib
import os
import pandas

def main():

    print('running')
    tvals = np.zeros((120,1));
    for i in range(10,130):
    	#tvals = np.concatenate((tvals,
        #	pandas.io.parsers.read_csv('/gpfs2/well/nichols/users/inf852/t'+ str(10*i) + '.csv')), axis=0)
        t = np.loadtxt('/gpfs2/well/nichols/users/inf852/t'+ str(10*i) + '.csv')
        print(t)
        print(t.shape)
        tvals[i-10] = t

    print(tvals)

if __name__ == "__main__":
    main()

