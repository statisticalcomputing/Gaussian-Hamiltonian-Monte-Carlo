''' stochastic ghmc
'''
import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
from argmin import argmin

from sqrtm import sqrtm,sqrtm_ad
#rs = npr.RandomState(0)
def ghmc(U,dU, dt = .000001,D=None, EPISODE=10000, BURNIN=None,VERBOSE=False,callback=None,POINTS=10,AC=0.5,STEPS=5):

    if D is None:
        print("D")
        exit(0)

    if BURNIN is None:
        BURNIN = int(EPISODE/2)

    n = POINTS

    z2x = lambda z: z
    x2z = lambda x: x

    K = lambda p: np.sum(p*p, axis=0)/2
    dK = lambda p: p

    pStar = np.random.randn(D, POINTS)
    xStar = np.random.randn(D, POINTS)

    cov_hat = np.eye(D)
    cov_hat_half = sqrtm(cov_hat)
    mu_0 = np.zeros((D,1))


    kappa_0 = n
    LAMBDA_0 = np.cov(xStar)

    H = np.sum(U(xStar) + K(pStar))

    p = [pStar]
    x = [xStar]
    fixed = False
    alphas = []
    deltas = []
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

        xStar = x[-1]
        K0 = H - U(xStar).sum()
        p0 = np.random.randn(D, POINTS)
        K1 = K(p0).sum()
        r0 = np.sqrt(K1/K0)
        pStar = p0 / r0

        zStar = x2z(xStar)
        E = [U(xStar)]
        for s in range(STEPS):
            zStar = zStar + dt*dK(pStar)
            pStar = pStar - dt*np.dot(cov_hat_half, dU(z2x(zStar)))
            E.append(U(z2x(zStar)))
        if VERBOSE:
            print(np.array(E).argmax(axis=0).mean(),np.array(E).argmin(axis=0).mean())
        xStar = z2x(zStar)

        alpha = np.exp(np.clip(U(x[-1])- U(xStar),-10,0))
        ridx = alpha < np.random.rand(POINTS)
        xStar[:,ridx] = x[-1][:,ridx]
        pStar[:,ridx] = p[-1][:,ridx]
        x.append(xStar)
        p.append(pStar)
        #alpha_bar = alpha_bar * 0.9 + alpha.mean()*.1
        
        M = np.array(E).argmax(axis=0)
        m = np.array(E).argmin(axis=0)
        Mi = np.logical_and(M>0 , M<STEPS-1)
        mi = np.logical_and(m>0 , m<STEPS-1)
        M_ = np.mean(M[Mi]) if M[Mi].size > 0 else np.nan
        m_ = np.mean(m[mi]) if m[mi].size > 0 else np.nan
        
        alphas.append(alpha.mean())
        deltas.append(dt)
        if j < BURNIN:
            if alpha.mean() > np.random.rand()*(1-0.3)+0.3:
                H = H*1.1
            elif alpha.mean() < np.random.rand()*0.3:
                H = H/1.1
            if np.isnan(M_) or np.isnan(m_):
                dt = dt*1.1
            elif np.isnan(M_) and np.isnan(m_):
                dt = dt/1.1
        if VERBOSE:
            print(j, np.mean(alpha),cov_hat,dt,np.cov(x[-1]),H,M_,m_)
    return {'x': np.swapaxes(np.array(x),1,2).reshape(-1,2), 'p':np.array(p),'alpha':alphas,'delta':deltas}


if __name__ == '__main__':
    dt = .000001
    POINTS = 100
    EPISODE = 10000
    BURNIN = int(EPISODE/2)
    R = 0.2
    D = 2
    n = POINTS
    STEPS = 5

    K = [.9, .99, .999, .9999, .99999, .999999, .9999999, .99999999, .999999999, .9999999999,]
    ds = []
    for i in range(len(K)):
        print(i)
        k = K[i]
        SIGMA = np.array([[1, k],[k, 1]])
        U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
        dU = lambda x: np.linalg.solve(SIGMA, x)

        np.random.seed(12345)
        info = ghmc(U, dU, D=D, dt=dt,EPISODE=EPISODE, POINTS=POINTS,VERBOSE=False)
        x = info['x']
        #np.cov(np.transpose(x[int(EPISODE/2):,:]))
        d0 = np.sum(np.square(np.cov(np.transpose(x[int(POINTS*EPISODE/2):,:])) - SIGMA))
        ds.append(d0)
    #plt.plot(x[:,0], x[:,1], '.b')
    #plt.pause(10);

    plt.plot(ds,'o-')
    plt.xlabel('k')
    plt.xticks(list(range(len(K))), [str(i) for i in K], rotation=20)

    plt.ylabel(r'$\left|\Sigma - \hat \Sigma\right|$')

    plt.show()
    plt.pause(10000)