import matplotlib.pyplot as plt
import numpy as np

tvals = np.loadtxt('tvals.csv', delimiter=',')
plt.plot(np.arange(10,1710,10),tvals)
plt.show()
