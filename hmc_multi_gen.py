'''hmc generator
'''
import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
import matplotlib.pyplot as plt

def hmc(U, dU, x0=None, dt = .0001, D=2, BURNIN=5000, VERBOSE=False, POINTS=10, AC=0.3, STEPS=10, bndchk=None):

    decay = 0.1
    decay_dt = 0.01
    DECAY_FACTOR = 1/exp(log(10000)/BURNIN)

    K = lambda p: np.sum(p*p, axis=0)/2
    dK = lambda p: p

    pStar = np.random.randn(D, POINTS)
    if x0 is None:
        xStar = np.random.randn(D, POINTS)
    else:
        xStar = x0

    H = np.sum(U(xStar) + K(pStar))

    x = xStar
    j = 0
    while True:
        xStar = x
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
        alpha = np.exp(np.clip(U(x)- U(xStar),-10,0))
        ridx = alpha < np.random.rand(POINTS)
        xStar[:,ridx] = x[:,ridx]
        if bndchk is not None:
            bndidx = bndchk(xStar)
            xStar[:,bndidx] = x[:,bndidx]
        yield xStar

        x = xStar

        #M = np.mean(np.array(E).argmax(axis=0))
        #m = np.mean(np.array(E).argmin(axis=0))
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
            print(j, np.mean(alpha),dt,M,m,S,s)
        j = j + 1


if __name__ == '__main__':
    np.random.seed(0)
    POINTS = 1000
    BURNIN = 5000
    D = 2
    SIGMA = np.array([[1, .9],[.9, 1]])
    U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
    dU = lambda x: np.linalg.solve(SIGMA, x)
    gen = hmc(U, dU, D=D, BURNIN=BURNIN, POINTS=POINTS)
    xs = []
    for j in range(10000):
        x = next(gen)
        if j>BURNIN:
            xs.append(x)

    print(np.cov(np.array(xs).reshape(-1,D).T))
