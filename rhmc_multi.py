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
HMC=1
RHMC=2
GHMC=3
def ghmc(U,dU, METHOD,dt = .000001,D=None, EPISODE=10000, BURNIN=None,VERBOSE=False,callback=None,POINTS=10,AC=0.5,STEPS=5,mu_0 = None, cov_0 = None, kappa_0=None):

    decay = 0.1
    decay_dt = 0.01
    if D is None:
        print("D")
        exit(0)

    if BURNIN is None:
        BURNIN = int(EPISODE/2)
    DECAY_FACTOR = 1/exp(log(10000)/BURNIN)
    n = POINTS

    z2x = lambda z: z
    x2z = lambda x: x

    if mu_0 is None:
        mu_0 = np.zeros((D,1))

    if cov_0 is None:
        cov_0 = np.eye(D)

    cov_hat_half = sqrtm(cov_0)
    z2x = lambda z: np.dot(cov_hat_half, z) + mu_0
    x2z = lambda x: np.linalg.solve(cov_hat_half, x - mu_0)


    K = lambda p: np.sum(p*p, axis=0)/2
    dK = lambda p: p

    pStar = np.random.randn(D, POINTS)
    xStar = np.random.randn(D, POINTS)

    if kappa_0 is None:
        kappa_0 = n
    LAMBDA_0 = 0#np.cov(xStar)

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

            if METHOD==RHMC:
                K = lambda p: np.sum(p * np.dot(cov_hat,p), axis = 0)/2
                dK = lambda p: np.dot(cov_hat, p)

        xStar = x[-1]
        K0 = np.clip(H - U(xStar).sum(),1e-3,H)
        p0 = np.random.randn(D, POINTS)
        K1 = K(p0).sum()
        r0 = np.sqrt(K1/K0)
        pStar = p0 / r0
        E = [U(xStar)]
        zStar = x2z(xStar)
        for s in range(STEPS):
            if METHOD == GHMC:
                zStar = zStar + dt*dK(pStar)
                pStar = pStar - dt*np.dot(cov_hat_half, dU(z2x(zStar)))
                E.append(U(z2x(zStar)))
                xStar = z2x(zStar)
            else:
                xStar = xStar + dt*dK(pStar)
                pStar = pStar - dt*dU(xStar)
                E.append(U(xStar))

        alpha = np.exp(np.clip(U(x[-1])- U(xStar),-10,0))
        ridx = alpha < np.random.rand(POINTS)
        xStar[:,ridx] = x[-1][:,ridx]
        pStar[:,ridx] = p[-1][:,ridx]
        x.append(xStar)
        p.append(pStar)

        M = np.mean(np.array(E).argmax(axis=0))
        m = np.mean(np.array(E).argmin(axis=0))
        S = np.unique(np.array(E).argmax(axis=0)).size
        s = np.unique(np.array(E).argmin(axis=0)).size

        alphas.append(alpha.mean())
        deltas.append(dt)
        if j < BURNIN:
            turned_AC = np.random.randn()*min(AC,1-AC)/6 + AC
            if alpha.mean() > turned_AC:
                H = H*(1+decay)
            elif alpha.mean() < turned_AC:
                H = H/(1+decay)
            if s==2 or S == 2:
                dt = dt*(1+decay_dt)
            elif S > 2:
            #elif np.array([M,m]).std()>1:
                dt = dt/(1+decay_dt)
            decay = decay * DECAY_FACTOR
            decay_dt = decay_dt * DECAY_FACTOR
        if VERBOSE and j % 1000 == 0:
            print(j, np.mean(alpha),dt,M,m,S,s,np.array([M,m]).std())
    return np.swapaxes(np.array(x),1,2).reshape(-1,D)


if __name__ == '__main__':
    #np.random.seed(0)
    dt = .0001
    POINTS = 1000
    EPISODE = 10000
    D = 2
    n = POINTS
    rho = [0.001,
           0.01,
           .1,
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
           .999999999,
    ]

    ds1 = []
    ds2 = []
    ds3 = []
    EXP = 2
    for i in range(len(rho)):
        r = rho[i]
        if EXP==1:
            SIGMA = np.array([[1, r],[r, 1]])
        else:
            SIGMA = np.array([[1/r, r],[r, 1/r]])
        U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
        dU = lambda x: np.linalg.solve(SIGMA, x)


        x = ghmc(U, dU, METHOD=HMC, D=D, dt=dt,EPISODE=EPISODE, POINTS=POINTS,VERBOSE=False,AC=0.3)

        d1 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(POINTS*EPISODE/2):,:])) - SIGMA)))
        ds1.append(d1)

        x = ghmc(U, dU, METHOD=RHMC, D=D, dt=dt,EPISODE=EPISODE, POINTS=POINTS,VERBOSE=False,AC=0.3)

        d2 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(POINTS*EPISODE/2):,:])) - SIGMA)))
        ds2.append(d2)

        x = ghmc(U, dU, METHOD=GHMC, D=D, dt=dt,EPISODE=EPISODE, POINTS=POINTS,VERBOSE=False,AC=0.3)

        d3 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(POINTS*EPISODE/2):,:])) - SIGMA)))
        ds3.append(d3)
        print(i, d1, d2, d3)

        [0,1.9409254850314897,1.1856478968328146,0.5080907617549059,
         1,0.1499106725319007,0.1611383055704225,0.16932057498266712,
         2,0.01212977526419503,0.011651004551217816,0.01504058752847049,
         3,0.010557082608917924,0.005547805154612504,0.005079961600129443,
         4,0.0019798044923549554,0.005136741394046156,0.0021140013767160905,
         5,0.0026926980517017534,0.007440241181122152,0.002864507701963361,
         6,0.003547239855934855,0.004183618844915349,0.0032736862382167034,
         7,0.0021072481339872978,0.005047465766708707,0.0015526806415814423,
         8,0.002621658629431255,0.004190912428106788,0.003362679473671474,
         9,0.0007654168463023596,0.0014077830110952511,0.0016627564005729772,
         10,0.0015797481611877077,0.002017420710246986,0.0018920044866341259,
         11,0.005318492619245797,0.0009901064270565088,0.002363692937605025,
         12,0.003468725687589862,9.089019509509227e-05,0.001472857047441415,
         13,0.07987122074707606,9.521113820382183e-05,0.0006016057246130336,
         14,0.2996464634234562,0.0009891314736658495,0.012561209462225569,
         15,0.34669293142090096,0.001814270476636424,0.010915785855743642,
         16,0.17981888210486896,0.001407703414708301,0.03222019521123948,
         17,0.7279028445835666,0.0004253575342902862,0.008148857683601065,]

    plt.plot(ds1,'x:')
    plt.plot(ds2,'o--')
    plt.plot(ds3,'*-')
    plt.yscale('log')
    plt.xlabel(r'$\rho$')
    plt.xticks(list(range(len(rho))), [str(r) for r in rho], rotation=20)

    plt.ylabel(r'$\left|\Sigma - \hat \Sigma\right|$')
    plt.legend(('HMC','RHMC','The proposed'))
    plt.show()
    plt.pause(10000)
