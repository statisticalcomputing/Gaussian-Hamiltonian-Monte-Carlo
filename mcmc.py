''' mcmc generator
'''
import numpy as np
from math import factorial
from numpy import log, inf
from scipy.special import gamma

def mcmc(U, x_min, x_max, x=None, POINTS = 10):
    if x is None:
        if not np.isinf(x_min) and not np.isinf(x_max):
            x = np.random.randint(x_min, x_max, POINTS)
        elif not np.isinf(x_min):
            x = np.random.randint(x_min, x_min+10, POINTS)
        elif not np.isinf(x_max):
            x = np.random.randint(x_max-10, x_max, POINTS)
    while True:
        if False:
            sgn = 2*np.random.randint(0,2,size=(POINTS))-1
            x_star = sgn + x
            fbd = np.logical_or(x_star < x_min , x_star >= x_max)
            x_star[fbd] = x[fbd]
            alpha = np.exp(np.clip(U(x) - U(x_star),-10,0))
            ridx = alpha < np.random.rand(POINTS)
            x_star[ridx] = x[ridx]
            yield x_star
        else:
            x_star = np.random.randint(x_min,x_max,size=(POINTS))
            alpha = np.exp(np.clip(U(x) - U(x_star),-10,0))
            ridx = alpha < np.random.rand(POINTS)
            x_star[ridx] = x[ridx]
            yield x_star
        x = x_star

if __name__ == '__main__':
    lam = 5
    POINTS = 1000
    U = lambda k: lam - k *log(lam) + log(gamma(k+1))
    gen = mcmc(U, x_min=0, x_max=inf, POINTS=POINTS)
    xs = []
    for i in range(1000):
        xs.append(next(gen))
    print(np.array(xs).mean())
