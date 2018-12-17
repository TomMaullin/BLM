import numpy as np
import yaml
import os
import local_blm_main

def main(i):
    
    inputsi = {'SVFlag': False,
               'contrasts': [{'c1': {'vector': [1], 
                                     'statType': 'T', 
                                     'name': 'contrast1'}}], 
               'outdir': '/well/nichols/users/inf852/' + str(i) + 'ukbbsv/', 
               'X': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/X_ukbb_' + str(i) + '.csv', 
               'Y_files': '/users/nichols/inf852/BLM-py/BLM/test/data/ukbb/Y_files_ukbb_' + str(i) + '.txt', 
               'MAXMEM': '2**31'}
    local_blm_main.main(inputsi)

if __name__ == "__main__":
    main()