import numpy as np
from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def sqrtm(x):
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    L = np.matmul(np.matmul(vh,np.diag(np.sqrt(np.abs(s)))),vh.T)
    return L

def hmc(U,dU, D,BURNIN,dt = .000001,PARTICLES=10,STEPS=10,AC=0.5, VANILLA=False):
    decay = 0.1
    decay_dt = 0.01
    DECAY_FACTOR = 1/exp(log(1000)/BURNIN)
    n = PARTICLES

    # bayesian
    mu_0 = np.zeros((D,1))
    cov_hat = np.eye(D)

    kappa_0 = n
    LAMBDA_0 = (n-1)*cov_hat

    # initial phase
    pStar = np.random.randn(D, PARTICLES)
    xStar = np.random.randn(D, PARTICLES)

    K = lambda p: np.sum(p * np.dot(cov_hat,p), axis = 0)/2

    # total energy
    H = np.sum(U(xStar) + K(pStar))
    j = 0
    while True:
        # bayesian estimation
        xx = xStar
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

        # kinetic
        K = lambda p: np.sum(p * np.dot(cov_hat,p), axis = 0)/2
        dK = lambda p: np.dot(cov_hat, p)

        # transformation
        z2x = lambda z: np.dot(cov_hat_half, z) + mu_0
        x2z = lambda x: np.linalg.solve(cov_hat_half, x - mu_0)
        dx2dz = lambda dx: np.linalg.solve(cov_hat_half, dx)

        p2b= lambda p: np.dot(cov_hat_half, p)
        b2p = lambda b: np.linalg.solve(cov_hat_half, b)
        dp2db= lambda dp: np.dot(cov_hat_half, dp)

        if VANILLA:
            pStar = np.random.randn(D, PARTICLES)
        else:
            # conserve total energy, i.e. equal probability jump
            K0 = np.clip(H - U(xStar).sum(),1e-3,H)
            p0 = np.random.randn(D, PARTICLES)
            K1 = K(p0).sum()
            r0 = np.sqrt(K0/K1)
            pStar = p0 * r0

        # previous coordinate
        _xStar = xStar
        _pStar = pStar

        E = [U(xStar)]

        # transform to new phase space
        zStar = x2z(xStar)
        bStar = p2b(pStar)

        # simulation in the new phase space
        for s in range(STEPS):
            zStar = zStar + dt*dx2dz(dK(b2p(bStar)))
            bStar = bStar - dt*dp2db(dU(z2x(zStar)))
            E.append(U(z2x(zStar)))

        # transform back to original phase space 
        xStar = z2x(zStar)
        pStar = b2p(bStar)

        # metropolis
        if VANILLA:
            alpha = np.exp(np.clip(U(_xStar) + K(_pStar) - U(xStar) - K(pStar),-20,0))
        else:
            alpha = np.exp(np.clip(U(_xStar)- U(xStar),-20,0))
        ridx = alpha < np.random.rand(PARTICLES)
        xStar[:,ridx] = _xStar[:,ridx]
        yield mu_0, cov_hat

        # extreme locations during simulation
        S = np.unique(np.array(E).argmax(axis=0)).size
        s = np.unique(np.array(E).argmin(axis=0)).size

        # tune total energy
        if not VANILLA:
            if alpha.mean() + alpha.std() > AC:
                H = H*(1+decay)
            elif alpha.mean() - alpha.std() < AC:
                H = H/(1+decay)

        # tune time step
        if j < BURNIN:
            if s==2 or S == 2:
                dt = dt*(1+decay_dt)
            elif S > 2:
                dt = dt/(1+decay_dt)
            decay_dt = decay_dt * DECAY_FACTOR
        j += 1

def hmc_vanilla(U,dU, D,BURNIN,dt = .0001,PARTICLES=10,STEPS=10,AC=0.3):
    decay_dt = 0.01

    #DECAY_FACTOR = 1/exp(log(1000)/BURNIN)

    # bayesian estimation
    n = PARTICLES
    mu_0 = np.zeros((D,1))
    cov_hat = np.eye(D)
    cov_hat_half = sqrtm(cov_hat)
    kappa_0 = n
    LAMBDA_0 = (n-1)*cov_hat

    # kinetic
    K = lambda p: np.sum(p*p, axis=0)/2
    dK = lambda p: p

    # initial phase
    pStar = np.random.randn(D, PARTICLES)
    xStar = np.random.randn(D, PARTICLES)

    j = 0
    while True:
        # bayesian estimation
        xx = xStar
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

        # regenerate momentum, not conserve energy
        pStar = np.random.randn(D, PARTICLES)
        # previous phase
        _pStar = pStar
        _xStar = xStar


        E = [U(xStar)]

        # Hamiltonian simulation
        for s in range(STEPS):
            xStar = xStar + dt*dK(pStar)
            pStar = pStar - dt*dU(xStar)
            E.append(U(xStar))

        # metropolis
        alpha = np.exp(np.clip(U(_xStar) + K(_pStar) - U(xStar) - K(pStar),-20,0))
        ridx = alpha < np.random.rand(PARTICLES)
        xStar[:,ridx] = _xStar[:,ridx]
        yield mu_0, cov_hat

        # extreme locations during simulation
        S = np.unique(np.array(E).argmax(axis=0)).size
        s = np.unique(np.array(E).argmin(axis=0)).size

        # tune time step, large error if stop tune after burnin
        if True:#j < BURNIN:
            if s==2 or S == 2:
                dt = dt*(1+decay_dt)
            elif S > 2:
                dt = dt/(1+decay_dt)
            #decay_dt = decay_dt * DECAY_FACTOR
        j += 1


if __name__ == '__main__':
    np.random.seed(0)
    PARTICLES = 300
    EPISODE = 30000
    BURNIN = 1000
    # Dimension
    DS = [2,3,6,10,20,30,40,50]
    
    rho = [.01,
           .1,
           .9,
           #.99,
           .999,
    ]

    EXP = 2
    DS1 = []
    DS2 = []
    DS_1 = []
    DS_2 = []
    color = ['red','black','blue','magenta','yellow']
    for i in range(len(DS)):
        ds1 = []
        ds2 = []
        ds_1 = []
        ds_2 = []
        D = DS[i]
        print("D",D)
        for j in range(len(rho)):
            r = rho[j]
            print("rho",r)
            MU = np.random.randn(D,1)
            SIGMA = np.ones((D,D))*r
            if EXP==1:
                np.fill_diagonal(SIGMA,1)
            else:
                np.fill_diagonal(SIGMA,1/r)
            U = lambda x: np.sum((x-MU) * np.linalg.solve(SIGMA,(x-MU)), axis = 0)/2
            dU = lambda x: np.linalg.solve(SIGMA, x-MU)

            gen = hmc_vanilla(U, dU, D=D, BURNIN=BURNIN,PARTICLES=PARTICLES)
            #gen = hmc(U, dU, D=D, BURNIN=BURNIN, PARTICLES=PARTICLES, VANILLA=True)
            for j in range(EPISODE):
                mu,cov = next(gen)
                # est. error of covariance
                d1 = np.sqrt(np.mean(np.square(cov - SIGMA)))
                # est. error of mean
                d_1 = np.sqrt(np.mean(np.square(mu-MU)))
                if j % 10000 == 0:
                    print("HMC",j, d1,d_1)
            ds1.append(d1)
            ds_1.append(d_1)

            gen = hmc(U, dU, D=D, BURNIN=BURNIN, PARTICLES=PARTICLES, VANILLA=False)
            for j in range(EPISODE):
                mu,cov = next(gen)
                d2 = np.sqrt(np.mean(np.square(cov - SIGMA)))
                d_2 = np.sqrt(np.mean(np.square(mu-MU)))
                if j % 10000 == 0:
                    print(j, d2,d_2)
            ds2.append(d2)
            ds_2.append(d_2)

        DS1.append(ds1)
        DS2.append(ds2)
        DS_1.append(ds_1)
        DS_2.append(ds_2)
    DS1 = np.array(DS1)
    DS2 = np.array(DS2)
    DS_1 = np.array(DS_1)
    DS_2 = np.array(DS_2)
    

    fontP = FontProperties()
    fontP.set_size('small')
    
    for i in range(len(rho)):
        plt.figure(1)
        plt.plot(DS1[:,i],'--',color=color[i],marker='*')
        plt.plot(DS2[:,i],'-' ,color=color[i],marker='+')
        plt.yscale('log')
        plt.xlabel(r'Dimension')
        plt.xticks(list(range(len(DS))), [str(d) for d in DS])

        plt.ylabel(r'$\left|\Sigma - \hat \Sigma\right|$')
        plt.legend([r'{}:$\rho$={}'.format(r,d) for d in rho for r in ['HMC','the proposed'] ],framealpha=0.5)#,fancybox=True,bbox_to_anchor=(1.12, 1))

        plt.figure(2)
        plt.plot(DS_1[:,i],'--',color=color[i],marker='*')
        plt.plot(DS_2[:,i],'-' ,color=color[i],marker='+')
        plt.yscale('log')
        plt.xlabel(r'Dimension')
        plt.xticks(list(range(len(DS))), [str(d) for d in DS])

        plt.ylabel(r'$\left|\mu - \hat \mu\right|$')
        plt.legend([r'{}:$\rho$={}'.format(r,d) for d in rho for r in ['HMC','the proposed'] ],framealpha=0.5)#,fancybox=True,bbox_to_anchor=(1.12, 1))
        #legend([plot1], "title", prop=fontP) 
    plt.show()
    plt.pause(1000000000)


