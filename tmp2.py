import numpy as np

j = 3000
X = np.ones([j,1])
np.savetxt(os.path.join(os.getcwd(),'BLM','test','data','ukbb', 'X_ukbb_' + str(1) + 'p.csv'), X, delimiter=",")
for i in range(2,11):

    X = np.concatenate((X,np.random.uniform(0, 1, [j, 1])), axis=1)
    np.savetxt(os.path.join(os.getcwd(),'BLM','test','data','ukbb', 'X_ukbb_' + str(i) + 'p.csv'), X, delimiter=",")

print('done')    
