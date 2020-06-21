''' "Massively Parallel" HMC with Coordinate Transformation
'''

import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt

def hmc(U,dU, dt = .000001,D=None, EPISODE=10000, BURNIN=None,VERBOSE=False,callback=None,POINTS=10,AC=0.5,STEPS=10):

    decay = 0.1
    decay_dt = 0.01

    if BURNIN is None:
        BURNIN = int(EPISODE/2)
    DECAY_FACTOR = 1/exp(log(10000)/BURNIN)
    n = POINTS

    K = lambda p: np.sum(p*p, axis=0)/2
    dK = lambda p: p

    pStar = np.random.randn(D, POINTS)
    xStar = np.random.randn(D, POINTS)

    H = np.sum(U(xStar) + K(pStar))

    x = [xStar]
    fixed = False
    for j in range(EPISODE):
        xStar = x[-1]
        K0 = np.clip(H - U(xStar).sum(),1e-3,H)
        p0 = np.random.randn(D, POINTS)
        K1 = K(p0).sum()
        r0 = np.sqrt(K1/K0)
        pStar = p0 / r0
        E = [U(xStar)]
        for s in range(STEPS):
            xStar = xStar + dt*dK(pStar)
            pStar = pStar - dt*dU(xStar)
            E.append(U(xStar))

        alpha = np.exp(np.clip(U(x[-1])- U(xStar),-10,0))
        ridx = alpha < np.random.rand(POINTS)
        xStar[:,ridx] = x[-1][:,ridx]
        x.append(xStar)

        S = np.unique(np.array(E).argmax(axis=0)).size
        s = np.unique(np.array(E).argmin(axis=0)).size

        if j < BURNIN:
            turned_AC = np.random.randn()*min(AC,1-AC)/6 + AC
            if alpha.mean() > turned_AC:
                H = H*(1+decay)
            elif alpha.mean() < turned_AC:
                H = H/(1+decay)
            if s==2 or S==2:
                dt = dt*(1+decay_dt)
            elif S > 2:
                dt = dt/(1+decay_dt)
            decay = decay * DECAY_FACTOR
            decay_dt = decay_dt * DECAY_FACTOR
        if VERBOSE and j % 100 == 0:
            print(j, np.mean(alpha),dt,S,s)
    return np.swapaxes(np.array(x),1,2).reshape(-1,D)


if __name__ == '__main__':
    np.random.seed(0)
    dt = .0001
    POINTS = 1000
    EPISODE = 10000
    D = 2
    n = POINTS
    rho = [.1,
           .2,
           .3,
           .4,
           .5,
           .6,
           .7,
           .8,
           .9,
           .99,
           .999,
           .9999,
           .99999,
           .999999,
           .9999999,
           .99999999,
    ]
    ds = []
    EXP = 2
    for i in range(len(rho)):
        print(i)
        r = rho[i]
        if EXP==1:
            SIGMA = np.array([[1, r],[r, 1]])
        else:
            SIGMA = np.array([[1/r, r],[r, 1/r]])
        U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
        dU = lambda x: np.linalg.solve(SIGMA, x)


        x = hmc(U, dU, D=D, dt=dt,EPISODE=EPISODE, POINTS=POINTS,VERBOSE=False,AC=0.3)

        d0 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(POINTS*EPISODE/2):,:])) - SIGMA)))
        ds.append(d0)
        print(np.cov(np.transpose(x[int(POINTS*EPISODE/2):,:])), SIGMA)
        print(d0)


    plt.plot(ds,'o-')
    plt.yscale('log')
    plt.xlabel(r'$\rho$')
    plt.xticks(list(range(len(rho))), [str(r) for r in rho], rotation=20)

    plt.ylabel(r'$\left|\Sigma - \hat \Sigma\right|$')

    plt.show()
    plt.pause(10000)
