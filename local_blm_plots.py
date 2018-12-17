import numpy as np
import yaml
import os
import local_blm_main
import time

def main(i):
    
    inputsi = {'SVFlag': True,
               'contrasts': [{'c1': {'vector': [1], 
                                     'statType': 'T', 
                                     'name': 'contrast1'}}], 
               'outdir': '/well/nichols/users/inf852/' + str(i) + 'ukbbsv/', 
               'X': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/X_ukbb_' + str(i) + '.csv', 
               'Y_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/Y_files_ukbb_' + str(i) + '.txt', 
               'MAXMEM': '2**31'}

    t1 = time.time()
    local_blm_main.main(inputsi)
    t2 = time.time()
    t=t2-t1

    np.savetxt('/well/nichols/users/inf852/t' + str(i) + '.csv',t, delimiter=",") 

if __name__ == "__main__":
    main()