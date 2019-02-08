import numpy as np
import os

def main():

    j = 200
    X = np.ones([j,1])
    for i in range(2,31):

        X = np.concatenate((X,np.random.uniform(0, 1, [j, 1])), axis=1)
        

    np.savetxt(os.path.join(os.getcwd(),'..','tmp2','X_200_p30.csv'), X, delimiter=",")

    print('done')    
