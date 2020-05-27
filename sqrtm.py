from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from optimizers import sgd,adam

#rs = npr.RandomState(0)

def sqrtm(x):
    w, v = np.linalg.eig(x+1e-8*np.eye(x.shape[0]))
    L = np.dot(np.dot(v, np.diag(np.sqrt(w))),np.transpose(v))
    #L = np.linalg.cholesky(x + 1e-15*np.eye(x.shape[0]))
    return L

def sqrtm_ad(A, x0=None):
    DIM = A.shape[0]
    def loss(x,t):
        y = x.reshape(DIM,DIM)
        loss = np.sum(np.abs(np.matmul(y,y.T).reshape(DIM*DIM) - A.reshape(DIM*DIM)))
        return loss
    if x0 is None:
        x0 = rs.randn(DIM*DIM)
    else:
        x0 = x0.reshape(DIM*DIM)
    def callback(params, t, g):
        #print("Iteration {} lower bound {}".format(t, loss(params, t)))
        if loss(params, t) < 1e-3:
            return True
        return False
    dloss = grad(loss)
    x = sgd(dloss, x0, step_size = .1,num_iters=10, callback=callback)
    z = x.reshape((DIM,DIM))
    return z
if __name__ == '__main__':
    DIM = 5
    A    = rs.randn(DIM,DIM)
    z = sqrtm(A)
    print(np.matmul(z,z))
    print(A)
