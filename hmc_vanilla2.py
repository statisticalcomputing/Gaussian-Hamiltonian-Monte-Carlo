import numpy as np
import matplotlib.pyplot as plt
HMC=1
CORRECTED = 2
CONSERVED = 3
def hmc(U, dU, x0= None,D=None, EPISODE=10000, BURNIN=None,VERBOSE=False,L=5, delta=.05,METHOD=HMC):

    if BURNIN is None:
        BURNIN = int(EPISODE/2)
    if METHOD==CONSERVED:
        decay = 0.1
        AC = 0.5
        DECAY_FACTOR = 1/np.exp(np.log(1000)/BURNIN)

    if x0 is None and D is None:
        print("either x0 or D")
        exit(0)
    if D is None:
        D = x0.size
    K = lambda p: np.dot(p, p)/2
    dK = lambda p: p
    if x0 is None:
        xStar = np.random.randn(D)
    else:
        xStar = x0
    pStar = np.random.randn(D)
    x  = [xStar]
    H = U(xStar) + K(pStar)
    for i in range(EPISODE):
        if METHOD==CONSERVED:
            xStar = x[-1]
            K0 = np.clip(H - U(xStar),1e-3,H)
            p1 = np.random.randn(D)
            #state s1
            K1 = K(p1)
            p2 = p1 / np.sqrt(K1/K0)
            pStar = p2
            #state s2 now
        else:
            # state s1
            x1 = x[-1]
            U_x1 = U(x1)
            p1 = np.random.randn(D)
            K_p1 = K(p1)

            p2 = np.random.randn(D)
            # state s2 now

            K_p2 = K(p2)
            H0 = U_x1 + K_p2

            pStar = p2
            xStar = x1

        for j in range(L):
            xStar = xStar + delta*dK(pStar)
            pStar = pStar - delta*dU(xStar)

        # state s3 now
        x2 = xStar
        p3 = pStar
        U_x2 = U(x2)
        K_p3 = K(p3)
        Hstar = U_x2 + K_p3

        if METHOD==HMC:
            alpha = np.exp(np.clip(H0-Hstar,-10,0))
        elif METHOD==CORRECTED:
            alpha = np.exp(np.clip(K_p2+U_x1-U_x2-K_p1,-10,0))
        elif METHOD==CONSERVED:
            alpha = np.exp(np.clip(U(x[-1])- U(xStar),-10,0))

        if alpha > np.random.rand():
            x.append(xStar)
        else:
            x.append(x[-1])
        if j < BURNIN and METHOD==CONSERVED:
            turned_AC = np.random.randn()*min(AC,1-AC)/6 + AC
            if alpha.mean() > turned_AC:
                H = H*(1+decay)
            elif alpha < turned_AC:
                H = H/(1+decay)
            decay = decay * DECAY_FACTOR
    return {'x': np.array(x)}


if __name__ == '__main__':
    np.random.seed(0)
    EPISODE = 10000
    D = 10
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

    EXP = 2
    if EXP == 1:
        DELTA = [0.0025,0.005,0.01,0.02, 0.04, 0.08]
        #DELTA = [0.08, 0.09,0.1, 0.11, 0.12, 0.13,0.14,0.15]
    else:
        #DELTA = [0.01,0.02, 0.04, 0.08, 0.16, 0.32]
        DELTA = [0.0025,0.005,0.01,0.02, 0.04, 0.08]
        #DELTA = [0.2,0.22,0.24,0.26,0.28]
    plt.ion()
    for i in range(len(DELTA)):
        delta = DELTA[i]
        ds = []
        ds1 = []
        ds2 = []
        print(delta)
        for j in range(len(rho)):
            r = rho[j]
            if EXP == 1:
                SIGMA = np.ones((D,D))*r
                np.fill_diagonal(SIGMA,1)
            else:
                SIGMA = np.ones((D,D))*r
                np.fill_diagonal(SIGMA,1/r)

            U = lambda x: np.sum(x * np.linalg.solve(SIGMA,x), axis = 0)/2
            dU = lambda x: np.linalg.solve(SIGMA, x)

            info = hmc(U, dU, D=D, delta=delta,EPISODE=EPISODE, METHOD=HMC)
            x = info['x']
            d0_ = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(EPISODE/2):,:])) - SIGMA)))
            ds.append(d0_)
            info = hmc(U, dU, D=D, delta=delta, EPISODE=EPISODE, METHOD=CORRECTED)
            x = info['x']
            d0 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(EPISODE/2):,:])) - SIGMA)))
            ds1.append(d0)

            info = hmc(U, dU, D=D, delta=delta, EPISODE=EPISODE, METHOD=CONSERVED)
            x = info['x']
            _d0 = np.sqrt(np.mean(np.square(np.cov(np.transpose(x[int(EPISODE/2):,:])) - SIGMA)))
            ds2.append(_d0)
            print(r, d0_, d0,_d0)
        plt.plot(ds,':')
        plt.plot(ds1,'--')
        plt.plot(ds2,'-')
        plt.yscale('log')
        plt.show()
        plt.pause(0.1)

    plt.xlabel(R'$\rho$')
    plt.xticks(list(range(len(rho))), [str(r) for r in rho])

    plt.ylabel(r'$\left|\Sigma - \hat \Sigma\right|$')
    plt.legend(['{}/{}'.format(r,d) for d in DELTA for r in ['hmc','corrected','conserved'] ])
    plt.show()
    plt.pause(10000)
