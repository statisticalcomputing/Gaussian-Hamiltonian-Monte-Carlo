import numpy as np
import matplotlib.pyplot as plt
from ghmc import ghmc

np.random.seed(54321)

D = 2
k1 = 70.
SIGMA = np.array([[1,k1/100.],[k1/100.,1]])
##SIGMA = np.array([[k1,1/k1],[1/k1,k1]])
U = lambda x: np.dot(x, np.dot(np.linalg.inv(SIGMA), x)) / 2
dU = lambda x: np.dot(np.linalg.inv(SIGMA),x)
if True:
    info = ghmc(U, dU, D,EPISODE=10000)
else:
    info = hmc(U, dU, D,EPISODE=1000, VERBOSE=True, delta = 0.5)

x = info['x']

print(np.cov(np.transpose(x[2000:,:])))

plt.plot(x[2000:,0],x[2000:,1],'.')
plt.show()
plt.pause(10)
