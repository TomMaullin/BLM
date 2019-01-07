import numpy as np
import pandas as pd
import os

def main():
    Y_files_file = '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/Y_files_ukbb_6063.txt'
    tsvdat = pd.read_csv('/well/nichols/projects/UKB/SMS/ukb25120-tfMRI-6063.tsv', delimiter='\t')

    keydat = pd.read_csv('/well/nichols/projects/UKB/SMS/IDs_tfMRI-6063-eid34077.tsv', delimiter='\t')
    print(keydat)
    print(tsvdat)

    with open(Y_files_file) as a:

        Y_files = []
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    X = np.zeros([len(Y_files)-1,3])
    i = 0
    j = 0
    Y_files2 = []
    while i < len(Y_files):

        Y_file = Y_files[i]
        Y_id_o = int(Y_file.replace('/gpfs2/well/nichols/projects/UKB/MNI/', '').replace('_tfMRI_cope1_MNI.nii.gz',''))

        if not keydat.loc[keydat['eid']==Y_id_o, 'eid34077'].empty:
            Y_id_n = int(keydat.loc[keydat['eid']==Y_id_o, 'eid34077'])
        else:
            Y_id_n = []

        if Y_id_n: #not tsvdat.loc[tsvdat['eid']==Y_id_n,'f.31.0.0'].empty:            

            Y_id_n = int(Y_id_n)

        else:
            
            print('Missing scan: ' + str(Y_id_o))

        
        if (Y_id_n) and (j < 200):
            X[j, 0] = 1
            Xj1 = tsvdat.loc[tsvdat['eid']==Y_id_n,'31-0.0'].values
            Xj2 = tsvdat.loc[tsvdat['eid']==Y_id_n,'34-0.0'].values
            X[j, 1] = Xj1
            X[j, 2] = Xj2
            j = j+1
            Y_files2.append(Y_file)

        i = i+1
            
    print(Y_files2)
    print(X)
    print(len(Y_files2))

    with open(os.path.join(os.getcwd(),'Y_200.txt'), 'a') as a:
        k = 0
        for Y_file2 in Y_files2:
            if k < 200:   
                a.write(Y_file2 + os.linesep)
                k = k + 1 

    with open(os.path.join(os.getcwd(),'M_200.txt'), 'a') as a:
        k = 0
        for Y_file2 in Y_files2:
            if k < 200:
                a.write(Y_file2.replace('cope1', 'mask') + os.linesep)
                k = k + 1

    np.savetxt(os.path.join(os.getcwd(),"X_200.csv"), X[0:200,:], delimiter=',') 
