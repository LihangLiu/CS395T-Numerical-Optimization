# min   c^T * x
# s.t.  Ax = b
#       x >= 0

import numpy as np
import itertools 

def init_A(m, n):
    return np.random.uniform(-1,1,[m,n])

def init_b(m):
    return np.random.uniform(-1,1,[m,1])

def init_c(n):
    return np.random.uniform(-1,1,[n,1])

def init_indice_B(A, b, m, n):
    for indice_B in list(itertools.combinations(range(n),m)):
        indice_B = np.array(indice_B)
        B = A[:, indice_B]
        x_B = np.linalg.solve(B, b)     # (m,1)
        if np.sum(x_B >= 0) != x_B.shape[0]:
            continue
        return indice_B

    print('no indice_B found')
    exit(0)

def complement(cur_indice, n):
    res = []
    for i in xrange(n):
        if i not in cur_indice:
            res.append(i)
    return np.array(res)

def objective_func(c, x):
    return np.matmul(c, x)

def simplex_method(A, b, c):
    m,n = A.shape
    indice_B = init_indice_B(A, b, m, n)  # (m)
    cnt = 0
    while True:
        B = A[:, indice_B]  # (m,m)
        indice_N = complement(indice_B, n) # (n-m)
        N = A[:, indice_N]  # (m,n-m)
        x_B = np.linalg.solve(B, b)     # (m,1)
        c_B = c[indice_B]   # (m,1)
        lamda = np.linalg.solve(B.T, c_B)   # (m,1)
        c_N = c[indice_N]   # (n-m,1)
        s_N = c_N - np.matmul(N.T, lamda)   # (n-m,1)

        if np.sum(s_N >= 0) == s_N.shape[0]:    # optimal points found
            break

        neg_indice = np.nonzero(s_N < 0)[0]
        q = indice_N[np.random.choice(neg_indice, 1)[0]]    # randomly select a q from negative s_N
        A_q = A[:, q]       # (m)
        d = np.linalg.solve(B, A_q)     # (m)
        if np.sum(d <= 0) == d.shape[0]:        # Unbounded
            print('Error: unbounded')
            exit(0)

        indice_nonzero_d = np.nonzero(d >= 0)[0]
        i = np.argmin(x_B[indice_nonzero_d].flatten()/d[indice_nonzero_d].flatten())

        indice_B = np.delete(indice_B, indice_nonzero_d[i])
        indice_B = np.append(indice_B, q)

        cnt += 1
    print('Iters used:', cnt)

    x = np.zeros([n,1])
    x[indice_B] = x_B
    return x

if __name__ == '__main__':
    np.random.seed(5)
    np.set_printoptions(precision=4, suppress=True)
    m = 10
    n = 20
    A = init_A(m, n)    # (m,n)
    b = init_b(m)    # (m,1)
    c = init_c(n)    # (n,1)

    print(simplex_method(A, b, c))







