import glob
import numpy as np
import nibabel as nib
import os
import pandas
import matplotlib as plt

def main():

    print('running')
    tvals = np.zeros((15,1));
    for i in range(0,15):

        filesi = glob.glob('/gpfs2/well/nichols/users/inf852/t_np'+ str(i+1) + '_*.csv')
        print(filesi)

        t = 0
        for file in filesi:

            #tvals = np.concatenate((tvals,
            #	pandas.io.parsers.read_csv('/gpfs2/well/nichols/users/inf852/t'+ str(10*i) + '.csv')), axis=0)
            print(pandas.io.parsers.read_csv(file,header=None))
            t = t + pandas.io.parsers.read_csv(file,header=None).values
            print(t)
            print(t.shape)
        
        tvals[i] = t/len(filesi)

    print(tvals)

    np.savetxt(os.path.join(os.getcwd(),"tparvals.csv"), 
               tvals, delimiter=",") 



if __name__ == "__main__":
    main()

