def argmin(U,dU, x0):
    print(x0)
    while True:
        x = x0 - 0.0001*dU(x0)
        if abs(U(x0)-U(x))< 1e-3:
            break
        x0 = x
    return x
