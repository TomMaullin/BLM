import numpy as np
import yaml
import os
import local_blm_main
import time
import shutil

def main(i,j):
    
    #inputsi = {'SVFlag': True,
    #           'contrasts': [{'c1': {'vector': [1], 
    #                                 'statType': 'T', 
    #                                 'name': 'contrast1'}}], 
    #           'outdir': '/well/nichols/users/inf852/' + str(i) + 'ukbbsv_' + str(j) + '/', 
    #           'X': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/X_ukbb_' + str(i) + '.csv', 
    #           'Y_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/Y_files_ukbb_' + str(i) + '.txt', 
    #           'M_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/M_files_ukbb_' + str(i) + '.txt',
    #           'MAXMEM': '2**31'}

    # inputsi = {'SVFlag': True,
    #            'contrasts': [{'c1': {'vector': [1], 
    #                                  'statType': 'T', 
    #                                  'name': 'contrast1'}}], 
    #            'outdir': '/well/nichols/users/inf852/3000ukbbsv_mem' + str(i) + '/', 
    #            'X': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/X_ukbb_3000_mem' + str(i) + '.csv', 
    #            'Y_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/Y_files_ukbb_3000_mem' + str(i) + '.txt', 
    #            'M_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/M_files_ukbb_3000_mem' + str(i) + '.txt',
    #            'MAXMEM': '2**' + str(i)}

    inputsi = {'SVFlag': True,
               'contrasts': [{'c1': {'vector': eval('[1' + (', 1'*(i-1)) + ']'), 
                                     'statType': 'T', 
                                     'name': 'contrast1'}}], 
               'outdir': '/well/nichols/users/inf852/3000ukbbsv_np' + str(i) + '_' + str(j) + '/', 
               'X': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/X_ukbb_' + str(i) + 'p.csv', 
               'Y_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/Y_files_ukbb_3000.txt', 
               'M_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/M_files_ukbb_3000.txt',
               'MAXMEM': '2**33'}

    t1 = time.time()
    local_blm_main.main(inputsi)
    t2 = time.time()
    t=t2-t1

    shutil.rmtree('/well/nichols/users/inf852/3000ukbbsv_np' + str(i)  + '_' + str(j) + '/')
    np.savetxt('/well/nichols/users/inf852/t_np' + str(i) + '_' + str(j) + '.csv',np.array([t]), delimiter=",") 
    
if __name__ == "__main__":
    main()
