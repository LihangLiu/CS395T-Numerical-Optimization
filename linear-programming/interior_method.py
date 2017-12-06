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

def init_x(n):
    return np.random.uniform(0,1,[n,1])

def init_s(n):
    return np.random.uniform(0,1,[n,1])

def init_lamda(m):
    return np.random.uniform(0,1,[m,1])

def get_J(A, S, X):
    m,n = A.shape
    J = np.zeros([2*n+m, 2*n+m])
    J[:n, n:n+m] = np.array(A.T)
    J[:n, n+m:] = np.identity(n)
    J[n:n+m, :n] = np.array(A)
    J[n+m:, :n] = np.array(S)
    J[n+m:, n+m:] = np.array(X)
    return J

def get_F(A, b, c, lamda, S, s, X, x, sigma, mu):
    m,n = A.shape
    res1 = np.matmul(A.T, lamda) + s - c
    res2 = np.matmul(A, x) - b
    e = np.ones([n,1])
    res3 = np.matmul(np.matmul(X, S), e) - sigma*mu*e
    return np.concatenate([res1, res2, res3], axis=0)

def get_alpha(v, delta_v):
    alphas = -v.flatten()/delta_v.flatten()
    alphas = alphas[alphas>0]
    return np.min(alphas)

def objective_func(c, x):
    return np.matmul(c.T, x)

def interior_method(A, b, c):
    m,n = A.shape
    x = init_x(n)   # (n,1)
    lamda = init_lamda(m)   # (m,1)
    s = init_s(n)   # (n,1)
    sigma0 = 0.5
    assert(np.sum(x>0) == n)
    assert(np.sum(s>0) == n)
    cnt = 0
    while True:
        sigma = sigma0 / (cnt+1)
        mu = np.matmul(x.T, s)/n
        S = np.diag(s.flatten())
        X = np.diag(x.flatten())
        J = get_J(A, S, X)
        F = get_F(A, b, c, lamda, S, s, X, x, sigma, mu)
        delta = np.linalg.solve(J, -F)
        delta_x, delta_lamda, delta_s = delta[:n], delta[n:n+m], delta[n+m:]
        alpha = get_alpha(np.append(x.flatten(), s.flatten()), np.append(delta_x.flatten(), delta_s.flatten()))
        x += alpha*delta_x
        lamda += alpha*delta_lamda
        s += alpha*delta_s

        if np.mean(np.abs(delta_x)) < 0.0001:
            break
        cnt += 1
    print('Iters used:', cnt)
    
    return x

if __name__ == '__main__':
    np.random.seed(5)
    np.set_printoptions(precision=4, suppress=True)
    m = 10
    n = 20
    A = init_A(m, n)    # (m,n)
    b = init_b(m)    # (m,1)
    c = init_c(n)    # (n,1)
    print(interior_method(A, b, c))






