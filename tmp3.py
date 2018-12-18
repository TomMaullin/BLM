import numpy as np
import pandas as pd

def main():
    Y_files = '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/Y_files_ukbb_6063.txt'
    X = np.zeros([len(Y_files),3])
    tsvdat = pd.read_csv('/well/nichols/projects/UKB/SMS/ukb24729-fMRI-6063-demo.tsv', delimiter='\t')

    print(Y_files)
    print(X)
    print(tsvdat)
    print(tsvdat.loc[tsvdat['eid']==5905827,'f.34.0.0'].values)

    for i in range(0, len(Y_files)):

        Y_file = Y_files[i]
        X[i, 1] = 1
        #X[i, 2] = 
    

    

        
