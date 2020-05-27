import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
from argmin import argmin
import gc
from math import log
def hmc(U, dU, x0= None,D=None, EPISODE=10000, BURNIN=None,VERBOSE=False,L=5, delta=.05,WRONG=False):

    
    if BURNIN is None:
        BURNIN = int(EPISODE/2)

    if x0 is None and D is None:
        print("either x0 or D")
        exit(0)
    if D is None:
        D = x0.size
    K = lambda p: np.dot(p, p)/2
    dK = lambda p: p
    if x0 is None:
        xStar = argmin(lambda x,_:U(x),lambda x,_:dU(x), np.random.randn(D))
    else:
        xStar = x0
    pStar = np.random.randn(D)
    x  = [xStar]
    for i in range(EPISODE):
        xStar = x[-1]
        K_p1 = K(pStar)
        pStar = np.random.randn(D)
        U_x1 = U(xStar)
        K_p2 = K(pStar)
        H0 = U(xStar) + K(pStar)
        for j in range(L):
            xStar = xStar + delta*dK(pStar)
            pStar = pStar - delta*dU(xStar)
        U_x2 = U(xStar)
        K_p3 = K(pStar)
        Hstar = U(xStar) + K(pStar)
        if WRONG:
            alpha = np.exp(np.clip(H0-Hstar,-10,0))
        else:
            alpha = np.exp(np.clip(K_p2+U_x1-U_x2-K_p1,-10,0))
        if alpha > np.random.rand():
            x.append(xStar)
        else:
            x.append(x[-1])
    return {'x': np.array(x)}


if __name__ == '__main__':
    EPISODE = 10000
    D = 2
    rho = [.1,
           .2,
           .3,
           .4,
           .5,
           .6,
           .7,
           .8,
           .9
    ]
    DELTA = [0.01,0.02, 0.04, 0.08, 0.16, 0.32]
    DS = []
    DS1 = []
    plt.ion()
    #np.random.seed(12345)
    for i in range(len(DELTA)):
        delta = DELTA[i]
        ds = []
        ds1 = []
        print(delta)
        for j in range(len(rho)):
            r = rho[j]
            #print('r',r)
            SIGMA = np.array([[1, r],[r, 1]])
            U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
            dU = lambda x: np.linalg.solve(SIGMA, x)

            info = hmc(U, dU, D=D, delta=delta,EPISODE=EPISODE, WRONG=True)
            x = info['x']
            d0_ = np.sum(np.square(np.cov(np.transpose(x[int(EPISODE/2):,:])) - SIGMA))
            ds.append(log(d0_))
            info = hmc(U, dU, D=D, delta=delta, EPISODE=EPISODE, WRONG=False)
            x = info['x']
            d0 = np.sum(np.square(np.cov(np.transpose(x[int(EPISODE/2):,:])) - SIGMA))
            ds1.append(log(d0))
            print(r, d0_, d0)
        DS.append(ds)
        DS1.append(ds1)
        plt.plot(ds,'--')
        plt.plot(ds1,'-')
        plt.show()
        plt.pause(0.1)
    # for i in range(len(DELTA)):
    #     plt.plot(DS[i],'--')
    #     plt.plot(DS1[i],'-')
    plt.xlabel('rho')
    plt.xticks(list(range(len(rho))), [str(r) for r in rho], rotation=90)

    plt.ylabel(r'log $\left|\Sigma - \hat \Sigma\right|$')
    plt.legend(['{}/{}'.format(r,d) for d in DELTA for r in ['hmc','corrected'] ])
    plt.show()
    plt.pause(10000)
