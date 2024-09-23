import numpy as np

def laplace_kernel(gamma=1, dx2dy=1):
    ''' Adjusted Laplacian kernel for different grid resolutions. '''

    ker = np.zeros((5, 5))
    d2_coeff = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
    ker[2, :] = d2_coeff / dx2dy
    ker[:, 2] += d2_coeff
    ker *= gamma
    ker[1:4, 1:4] += np.array([[1, 0, 1],
                               [0, -4, 0],
                               [1, 0, 1]])\
                     / np.sqrt(1+dx2dy**2)
    return ker