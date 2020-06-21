''' "Massively Parallel" HMC with Coordinate Transformation
'''

import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt



def sqrtm(x):
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    L = np.matmul(np.matmul(vh,np.diag(np.sqrt(np.abs(s)))),vh.T)
    return L

def ghmc(U,dU, D,dt = .000001,EPISODE=10000, BURNIN=None,VERBOSE=False,POINTS=10,AC=0.3,STEPS=10):

    decay = 0.1
    decay_dt = 0.01

    if BURNIN is None:
        BURNIN = int(EPISODE/4)
    DECAY_FACTOR = 1/exp(log(1000)/BURNIN)
    n = POINTS

    z2x = lambda z: z
    x2z = lambda x: x

    mu_0 = np.zeros((D,1))
    cov_0 = np.eye(D)

    cov_hat_half = sqrtm(cov_0)
    z2x = lambda z: np.dot(cov_hat_half, z) + mu_0
    x2z = lambda x: np.linalg.solve(cov_hat_half, x - mu_0)


    K = lambda p: np.sum(p*p, axis=0)/2
    dK = lambda p: p

    pStar = np.random.randn(D, POINTS)
    xStar = np.random.randn(D, POINTS)

    kappa_0 = n
    LAMBDA_0 = 0

    H = np.sum(U(xStar) + K(pStar))

    p = [pStar]
    x = [xStar]

    for j in range(EPISODE):
        if j<BURNIN:
            xx = x[-1]
            x_bar = np.mean(xx,axis = 1,keepdims=True)
            kappa_n = kappa_0 + n
            mu_n = (kappa_0 * mu_0 + n * x_bar)/kappa_n

            S = (n-1) * np.cov(xx)
            LAMBDA_n = LAMBDA_0 + S + kappa_0 * n / (kappa_0 + n) * np.outer(x_bar - mu_0, x_bar - mu_0)
            cov_hat = LAMBDA_n/(kappa_0+n-D-1)
            cov_hat_half = sqrtm(cov_hat)
            mu_0 = mu_n
            kappa_0 = kappa_n
            LAMBDA_0 = LAMBDA_n
            z2x = lambda z: np.dot(cov_hat_half, z) + mu_0
            x2z = lambda x: np.linalg.solve(cov_hat_half, x - mu_0)
            K = lambda p: np.sum(p * np.dot(cov_hat,p), axis = 0)/2
            dK = lambda p: np.dot(cov_hat, p)

        xStar = x[-1]
        K0 = np.clip(H - U(xStar).sum(),1e-3,H)
        p0 = np.random.randn(D, POINTS)
        K1 = K(p0).sum()
        r0 = np.sqrt(K1/K0)
        pStar = p0 / r0
        p1 = pStar.copy()
        x1 = xStar

        E = [U(xStar)]
        for s in range(STEPS):
            xStar = xStar + dt*dK(pStar)
            pStar = pStar - dt*dU(xStar)
            E.append(U(xStar))


        alpha = np.exp(np.clip(U(x[-1])- U(xStar),-10,0))

        ridx = alpha < np.random.rand(POINTS)
        xStar[:,ridx] = x[-1][:,ridx]
        pStar[:,ridx] = p[-1][:,ridx]
        x.append(xStar)
        p.append(pStar)

        S = np.unique(np.array(E).argmax(axis=0)).size
        s = np.unique(np.array(E).argmin(axis=0)).size

        if j < BURNIN:
            turned_AC = np.random.randn()*min(AC,1-AC)/6 + AC
            if alpha.mean() > turned_AC:
                H = H*(1+decay)
            elif alpha.mean() < turned_AC:
                H = H/(1+decay)
            if s==2 or S == 2:
                dt = dt*(1+decay_dt)
            elif S > 2:
                dt = dt/(1+decay_dt)
            decay = decay * DECAY_FACTOR
            decay_dt = decay_dt * DECAY_FACTOR
        if VERBOSE and j % 1000 == 0:
            print(j, np.mean(alpha),dt,S,s)
    return np.swapaxes(np.array(x),1,2).reshape(-1,D)


if __name__ == '__main__':
    #np.random.seed(0)
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
    EXP = 1
    for i in range(len(rho)):
        print(i)
        r = rho[i]
        if EXP==1:
            SIGMA = np.array([[1, r],[r, 1]])
        else:
            SIGMA = np.array([[1/r, r],[r, 1/r]])
        U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
        dU = lambda x: np.linalg.solve(SIGMA, x)


        x = ghmc(U, dU, D=D, dt=dt,EPISODE=EPISODE, POINTS=POINTS,VERBOSE=False,AC=0.3)
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
