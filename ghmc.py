# workable conserved hmc
import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
from argmin import argmin
import scipy as sci

def sqrtm(x):
    w, v = np.linalg.eig(x)
    return np.dot(np.dot(v, np.diag(np.sqrt(w))),np.transpose(v))
 
def ghmc(U,dU, D, x0= None, EPISODE=10000, BURNIN0=100, BURNIN1=2000,BURNIN2=4000,L=5,decay=0.1, delta=0.001,n = 30,VERBOSE=False):
    DECAY_FACTOR = 1/exp(log(1000)/BURNIN2);
    K = lambda p: np.dot(p, p)/2
    dK = lambda p: p
    if x0 == None:
        x0 = argmin(U,dU, np.random.randn(D))
    p0 = np.random.randn(D)
    x  = [x0]
    p  = [p0]
    E_total = U(x0) + K(p0)
    z2x = lambda z: z
    x2z = lambda x: x
    rho = .5
    cov_hat = np.eye(D)
    cov_hat_half = cov_hat
    alphas = []
    zcs = []
    deltas = []
    Es = []
    Us = []
    Ks = []
    for i in range(EPISODE):
        xStar = x[-1]
        if E_total > U(xStar):
            if i == BURNIN0:
                ## bayesian
                mu_0 = np.mean(np.array(x),axis = 0)
                kappa_0 = BURNIN0
                LAMBDA_0 = np.cov(np.transpose(x))
                nu_0 = BURNIN0
            elif i > BURNIN0  and i%n == 0 and i < BURNIN1:
                y = x[-n:]
                ## bayesian
                y_bar = np.mean(np.array(y),axis = 0)
                S = (n-1) * np.cov(np.transpose(y))
                mu_n = kappa_0 * mu_0 / (kappa_0 + n) + n / (kappa_0 + n) * y_bar
                kappa_n = kappa_0 + n
                nu_n = nu_0 + n
                LAMBDA_n = LAMBDA_0 + S + kappa_0 * n / (kappa_0 + n) * np.outer(y_bar - mu_0, y_bar - mu_0)

                cov_hat = LAMBDA_n/(nu_0+n-D-1)
                #fprintf('%f\t%f\n',norm(cov_hat - SIGMA),norm(mu_0))
                mu_0 = mu_n
                kappa_0 = kappa_n
                nu_0 = nu_n
                LAMBDA_0 = LAMBDA_n
                cov_hat_half = sqrtm(cov_hat)
                z2x = lambda z: np.dot(cov_hat_half, z) + mu_0
                x2z = lambda x: np.dot(np.linalg.inv(cov_hat_half), x - mu_0)
                ##dst_sigma(:,end+1) = [i norm(cov_hat - SIGMA)];
                ##dst_mu(:,end+1) = [i norm(mu_0)];
            R = np.sqrt((E_total - U(xStar))*2)
        else:
            R = 1e-6
        p0 = np.random.randn(D)
        p0 = p0 / np.linalg.norm(p0)
        pStar = p0 * R
        E_total = U(xStar) + K(pStar)
        E_t = [K(pStar)]
        zc = 0
        zStar = x2z(xStar)
        pStar = pStar - .5 * delta*dU(xStar)
        for j in range(1, L+1):
            zStar = zStar + delta*dK(pStar)
            pStar = pStar - delta*np.dot(cov_hat_half, dU(z2x(zStar)))
            E_t.append(K(pStar))
            if j >= 2 and zc == 0:
                if (E_t[-3] - E_t[-2])*(E_t[-2] - E_t[-1])<0:
                    zc = j
        xStar = z2x(zStar)
        pStar = pStar + .5 * delta*dU(xStar)
        if E_total < U(xStar):
            x.append(x[-1])
            p.append(p[-1])
        else:
            alpha = min(1,exp(U(x[-1])- U(xStar)))
            u = rand()
            if u > alpha:
                x.append(x[-1])
                p.append(p[-1])
            else:
                x.append(xStar)
                p.append(pStar)
            if i < BURNIN2:
                if rand() < rho:
                    if zc == 0 or zc == L:
                        if zc==L and  alpha < rand()*.3:
                            delta = delta / (1. + decay)
                        elif zc==0 and alpha > rand()*.7 + .3:
                            delta = delta * (1. + decay)
                else:
                    if alpha > rand()*.7 + .3:
                        E_total = E_total * (1. + decay)
                    elif alpha < rand()*.3:
                        E_total = E_total / (1. + decay)
                decay = decay * DECAY_FACTOR
                rho = rho * DECAY_FACTOR
            if VERBOSE:
                if i == 0:
                    print('idx\talpha\tzc\tdelta\tE_total\t\tU\t\tK');
                print('{}\t{:.3f}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(i,alpha,zc,delta,E_total,U(x[-1]),K(p[-1])))
            #else:
                #print('.',end='')
            alphas.append(alpha)
            zcs.append(zc)
            deltas.append(delta)
            Es.append(E_total)
            Us.append(U(x[-1]))
            Ks.append(K(p[-1]))
    print('')
    return {'x': np.array(x), 'p':np.array(p),'alpha':alphas,'zc':zcs,'delta':deltas,'H':Es,'U':Us,'K':Ks}


if __name__ == '__main__':
    k1=.9999;
    SIGMA=np.array([[1,k1], [k1,1]])
    ISIGMA = np.linalg.inv(SIGMA)
    U = lambda x: np.dot(np.dot(ISIGMA, x), x)/2
    dU = lambda x: np.dot(ISIGMA, x)
    np.random.seed(54321)
    info = ghmc(U, dU, 2)
    x = info['x']
    print(np.cov(np.transpose(x[5000:,:])))
    plt.plot(x[:,0], x[:,1], '.b')
    plt.pause(10);
