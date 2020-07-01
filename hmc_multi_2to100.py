'''
compare HMC with CHMC or HMC with HMC_NEW
simultaneously sample multiple particles (Markov chains)
redistribute energy among particles (the conserve energy method)
'''

import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
import gc

HMC    = 1

# No conserve energy, only new acceptance probability (AP1)
HMC_NEW = 2

# Conserve energy, also new acceptance probability (AP2)
CHMC   = 3

def hmc(U,dU, METHOD,D, dt = .0001, EPISODE=10000, BURNIN=None, VERBOSE=False, PARTICLES=10, AC=0.3, STEPS=5):

    decay = 0.1
    decay_dt = 0.01

    # in BURNIN period, tune time step and total energy (conserve energy method)
    if BURNIN is None:
        BURNIN = int(EPISODE/2)

    # update strength decayed to 1/1000 at the end of BURNIN period
    DECAY_FACTOR = 1/exp(log(1000)/BURNIN)

    # kinetic energy
    K = lambda p: np.sum(p*p, axis=0)/2
    dK = lambda p: p

    # init coordinate and momentum
    pStar = np.random.randn(D, PARTICLES)
    xStar = np.random.randn(D, PARTICLES)

    # total energy, will be updated according to acceptance probability
    H = U(xStar).sum() + K(pStar).sum()
    delta_H  = 1

    # samples
    x = [xStar]

    for i in range(EPISODE):
        xStar = x[-1]

        if METHOD == CHMC:
            # conserve energy method
            K0 = np.clip(H - U(xStar).sum(),1e-3, H)
            p0 = np.random.randn(D, PARTICLES)
            K1 = K(p0).sum()
            r0 = np.sqrt(K0/K1)
            pStar = p0 * r0
        elif METHOD==HMC:
            # vanilla HMC
            pStar = np.random.randn(D, PARTICLES)
            _pStar = pStar
        elif METHOD==HMC_NEW:
            # new HMC method
            # state s1
            # x1
            U_x1 = U(xStar)
            # using momentum of previous hamiltonian simulation
            # p1
            K_p1 = K(pStar)

            # p2
            pStar = np.random.randn(D, PARTICLES)
            # state s2 now

            K_p2 = K(pStar)


        # record potential energies during hamiltonian simulation, using the extreme of energies for finding time step
        E = [U(xStar)]
        for s in range(STEPS):
            xStar = xStar + dt*dK(pStar)
            pStar = pStar - dt*dU(xStar)
            E.append(U(xStar))

        if METHOD == CHMC:
            #AP2
            alpha = np.exp(np.clip(U(x[-1])- U(xStar),-20,0))

        elif METHOD==HMC:
            #p3
            #K_p3 = K(pStar)
            alpha = np.exp(np.clip(U(x[-1]) + K(_pStar) - U(xStar) - K(pStar),-20,0))

        elif METHOD==HMC_NEW:
            # x2
            U_x2 = U(xStar)

            #AP1
            alpha = np.exp(np.clip(K_p2+U_x1-U_x2-K_p1,-20,0))

        # MH samping
        ridx = alpha < np.random.rand(PARTICLES)
        xStar[:,ridx] = x[-1][:,ridx]
        x.append(xStar)

        # if time step is too small, S (and s) will be 2.
        S = np.unique(np.array(E).argmax(axis=0)).size
        s = np.unique(np.array(E).argmin(axis=0)).size

        if i < BURNIN:
            # high acceptance probability: decrease total energy
            if alpha.mean() + alpha.std()*3 > AC:
                H = H +  delta_H
            elif alpha.mean() - alpha.std()*3 < AC:
                H = max(H -  delta_H, 1e-3)
            if s==2 or S == 2:
                dt = dt*(1+decay_dt)
            elif S > 2:
                dt = dt/(1+decay_dt)
            decay_dt = decay_dt * DECAY_FACTOR
    return np.swapaxes(np.array(x),1,2).reshape(-1,D)


if __name__ == '__main__':
    np.random.seed(12345)
    PARTICLES = 30
    EPISODE = 10000
    DS = [2,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
    rho = [0.01,
           .1,
           .5]
    # 1: HMC VS HMC_NEW, 2: HMC vs Conserved HMC
    EXP = 2
    DS1 = []
    DS2 = []
    for i in range(len(DS)):
        ds1 = []
        ds2 = []
        D = DS[i]
        for j in range(len(rho)):
            r = rho[j]
            SIGMA = np.ones((D,D))*r
            np.fill_diagonal(SIGMA,1/r)
            U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
            dU = lambda x: np.linalg.solve(SIGMA, x)

            x = hmc(U, dU, METHOD=HMC, D=D, EPISODE=EPISODE, PARTICLES=PARTICLES)
            d1 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(PARTICLES*EPISODE/2):,:])) - SIGMA)))
            ds1.append(d1)
            if EXP==1:
                x = hmc(U, dU, METHOD=HMC_NEW, D=D, EPISODE=EPISODE, PARTICLES=PARTICLES)
            elif EXP==2:
                x = hmc(U, dU, METHOD=CHMC, D=D, EPISODE=EPISODE, PARTICLES=PARTICLES)
            d2 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(PARTICLES*EPISODE/2):,:])) - SIGMA)))
            ds2.append(d2)
            print(D, d1,d2)
            gc.collect()

        DS1.append(ds1)
        DS2.append(ds2)

    DS1 = np.array(DS1)
    DS2 = np.array(DS2)
    for i in range(len(rho)):
        plt.plot(DS1[:,i],'--',color='red',marker='*')
        plt.plot(DS2[:,i],'-' ,color='black',marker='+')
    plt.yscale('log')
    plt.xlabel(r'Dimension')
    plt.xticks(list(range(len(DS))), [str(d) for d in DS])

    plt.ylabel(r'$\left|\Sigma - \hat \Sigma\right|$')
    if EXP==1:
        plt.legend([r'{}:$\rho$={}'.format(r,d) for d in rho for r in ['hmc','the proposed'] ])
    elif EXP==2:
        plt.legend([r'{}:$\rho$={}'.format(r,d) for d in rho for r in ['hmc','conserve energy'] ])
    plt.show()
    plt.pause(10000)
