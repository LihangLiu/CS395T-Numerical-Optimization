import numpy as np

from interior_method import interior_method
from simplex_method import simplex_method

def init_A(m, n):
    # return np.array([[1,1,1,0],
    #                 [2,0.5,0,1]])
    return np.random.uniform(-1,1,[m,n])

def init_b(m):
    # return np.array([[5],
    #                 [8]])
    return np.random.uniform(-1,1,[m,1])

def init_c(n):
    # return np.array([[-3],[-2],[0],[0]])
    return np.random.uniform(-1,1,[n,1])

def init_x(n):
    return np.random.uniform(0,1,[n,1])

if __name__ == '__main__':
    # np.random.seed(1)
    np.set_printoptions(precision=4, suppress=True)
    m = 10
    n = 20
    A = init_A(m, n)    # (m,n)
    b = np.matmul(A, init_x(n))    # (m,1)
    c = init_c(n)    # (n,1)
    # print(c[:10])

    # run simplex method
    print('----------')
    print('Running simplex method ...')
    res_sim = simplex_method(A, b, c)
    print('x*:')
    print(res_sim.flatten())

    # run interior point method
    print('----------')
    print('Running interior point method ...')
    res_int = interior_method(A, b, c)
    print('x*:')
    print(res_int.flatten())

    
