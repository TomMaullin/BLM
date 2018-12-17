import glob
import numpy as np
import nibabel as nib
import os
import pandas
import matplotlib as plt

def main():

    print('running')
    tvals = np.zeros((130,1));
    for i in range(10,140):
    	#tvals = np.concatenate((tvals,
        #	pandas.io.parsers.read_csv('/gpfs2/well/nichols/users/inf852/t'+ str(10*i) + '.csv')), axis=0)
        t = np.loadtxt('/gpfs2/well/nichols/users/inf852/t'+ str(10*i) + '.csv')
        print(t)
        print(t.shape)
        tvals[i-10] = t

    np.savetxt(os.path.join(os.getcwd(),"tvals.csv"), 
               t, delimiter=",") 



if __name__ == "__main__":
    main()

