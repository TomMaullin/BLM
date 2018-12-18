import matplotlib.pyplot as plt
import numpy as np

tvals = np.loadtxt('tvals1.csv', delimiter=',')
for i in range(1,16):
    tvals = tvals + np.loadtxt('tvals' + str(i) + '.csv', delimiter=',')
print(len(np.arange(10,2920,10)))
print(len(tvals))
fig, ax = plt.subplots()
ax.set_xlabel('Number of Subjects')
ax.set_ylabel('Time (seconds)')
plt.plot(np.arange(10,2920,10),tvals/15)

np.savetxt('tvals_total_ns.csv', tvals/15)
plt.show()
