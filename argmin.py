from optimizers import sgd,adam

def argmin(U,dU, x0):
    def callback(params, t, g):
        if U(params, t) < 1e-3:
            return True
        return False
    x = adam(dU, x0, step_size = .01,num_iters=1000, callback=callback)
    return x

