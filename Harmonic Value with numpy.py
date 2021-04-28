import numpy as np

def harmonic(n):
    x = np.arange(1,n+1,2)
    y = np.arange(2,n+1,2)
    new_x = 1 / x
    new_y = -1 / y
    return np.sum(new_x) + np.sum(new_y)
