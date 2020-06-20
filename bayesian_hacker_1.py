from mcmc import mcmc
from hmc_multi_gen import hmc
from numpy import loadtxt, inf, log
from numpy.random import exponential, randint
import autograd.numpy as np
from scipy.special import gamma
from autograd import elementwise_grad as grad
import matplotlib.pyplot as plt

POINTS = 100
D = 2
BURNIN = 5000

# data

count_data = loadtxt("/home/user/Downloads/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers-master/Chapter1_Introduction/data/txtdata.csv")
n_count_data = len(count_data)
count_data = np.array(count_data)
count_data = np.repeat(count_data[np.newaxis,:], POINTS, axis=0)

# hyper param
alpha = 1.0/count_data.mean()

# init params
lam0 = np.vstack((exponential(scale=1/alpha,size=POINTS),
                       exponential(scale=1/alpha,size=POINTS)))

tau0  = randint(0, n_count_data, POINTS)


# energy
poisson_U = lambda lam: lam - count_data * np.log(lam) + log(gamma(count_data+1))
exp_U = lambda z: alpha * z - np.log(alpha)

def lambda_(tau, lam1, lam2):
    out = np.zeros((POINTS, n_count_data)) # number of data points
    out = []
    for i in range(POINTS):
        out.append(np.concatenate((np.ones(tau[i]) * lam1[i],np.ones(n_count_data-tau[i]) * lam2[i])))
    return np.array(out)

def U(x):
    global tau0
    lam1, lam2 = x[0,:], x[1,:]
    lam = lambda_(tau0, lam1, lam2)
    return np.sum(poisson_U(lam),axis=1) + exp_U(lam1) + exp_U(lam2)

dU = grad(U)

def V(tau):
    global lam0
    lam1, lam2 = lam0[0,:], lam0[1,:]
    lam = lambda_(tau, lam1, lam2)
    return np.sum(poisson_U(lam),axis=1) + exp_U(lam1) + exp_U(lam2)
bndchk = lambda x:np.any(x<0,axis=0)
gen_V = mcmc(V, x_min=0, x_max=n_count_data, POINTS=POINTS)
gen_U = hmc(U, dU, x0=lam0, D=D, BURNIN=BURNIN, POINTS=POINTS,bndchk=bndchk)
TAU = []
LAM0 = []
LAM1 = []
for i in range(BURNIN*2):
    tau0 = next(gen_V)
    lam0 = next(gen_U)
    print(i)
    if i > BURNIN:
        TAU.append(tau0)
        LAM0.append(lam0[0,:])
        LAM1.append(lam0[1,:])
TAU = np.array(TAU).reshape(-1)
LAM0 = np.array(LAM0).reshape(-1)
LAM1 = np.array(LAM1).reshape(-1)

plt.figure()
plt.scatter(LAM0,LAM1)
plt.xlabel("lam0")
plt.ylabel("lam1")
plt.figure()


# histogram of the samples
ax = plt.subplot(311)
ax.set_autoscaley_on(False)
plt.hist(LAM0, histtype='stepfilled', bins=100, alpha=0.85, label="posterior of $\lambda_1$", color="#A60628",density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the parameters $\lambda_1,\;\lambda_2,\;\tau$""")
#plt.xlim([15, 30])
plt.ylim([0, .6])
plt.xlabel("$\lambda_1$ value")
plt.ylabel("Density")
ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(LAM1, histtype='stepfilled', bins=100, alpha=0.85, label="posterior of $\lambda_2$", color="#7A68A6",density=True)
plt.legend(loc="upper left")
#plt.xlim([15, 30])
plt.ylim([0, .6])
plt.xlabel("$\lambda_2$ value")
plt.ylabel("Density")
plt.subplot(313)
w = 1.0 / TAU.shape[0] * np.ones_like(TAU)
plt.hist(TAU, bins=n_count_data, alpha=1, label=r"posterior of $\tau$", color="#467821", rwidth=2. ,density=True)
plt.xticks(np.arange(n_count_data))
plt.legend(loc="upper left")
plt.ylim([0, .6])
#plt.xlim([35, 54])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("Probability");



plt.show()
plt.pause(10000)
