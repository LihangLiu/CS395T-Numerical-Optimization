import numpy as np
from numpy import linalg

###################
# util functions
###################

def init_G():
    return np.random.binomial(1, 0.3, size=[N,N])
    # test
    # return np.ones([N,N])

def init_A():
    return np.random.uniform(0, 1, size=[N,N])
    # test
    # return np.ones([N,N])

def init_x():
    return np.random.normal(0, 1, size=[N*R*2])    
    # test
    # return np.ones([N*R*2])

def BC_2_x(B, C):
    return np.append(B.reshape([N*R]), C.reshape([N*R]))

def x_2_BC(x):
    return x[:N*R].reshape([N,R]), x[N*R:].reshape([R,N])

def func_diff_d_b(i, j, k, l):
    if k == i:
        return C[l, j]
    return 0

def func_diff_d_c(i, j, m, n):
    if n == j:
        return B[i, m]
    return 0

def func_diff_d_b_c(i, j, k, l, m, n):
    if k==i and m==l and n==j:
        return 1
    return 0

###################
# diff functions
###################

def func_diff_f_b(k, l):
    res = 0
    D = np.matmul(B, C)
    for i in xrange(N):
        for j in xrange(N):
            res += G[i,j]*2*(A[i,j]-D[i,j]) * (-func_diff_d_b(i,j,k,l))
    res += mu*B[k,l]

    return res

def func_diff_f_c(m, n):
    res = 0
    D = np.matmul(B, C)
    for i in xrange(N):
        for j in xrange(N):
            res += G[i,j]*2*(A[i,j]-D[i,j]) * (-func_diff_d_c(i,j,m,n))
    res += mu*C[m,n]

    return res


###################
# f, g, H
###################

def func_f(x):
    res = 0
    B, C = x_2_BC(x)
    D = np.matmul(B, C)
    for i in xrange(N):
        for j in xrange(N):
            res += G[i,j]*(A[i,j]-D[i,j])**2
    res += 0.5*mu*(np.sum(np.square(B)) + np.sum(np.square(C)))
    return res

def func_g():
    g = []
    for k in xrange(N):
        for l in xrange(R):
            g.append(func_diff_f_b(k, l))

    for m in xrange(R):
        for n in xrange(N):
            g.append(func_diff_f_c(m, n))

    g = np.array(g)
    return g


###################
# trust region subproblem
###################


if __name__ == '__main__':
    N = 10
    R = 2
    mu = 0.1
    np.random.seed(21)
    A = init_A()
    G = init_G()
    alpha = 0.01
    eta = 0.2
    x = init_x()
    cnt = 0
    while True:
        B, C = x_2_BC(x)
        g = func_g()
        x = x - alpha*g

        f_x = func_f(x)
        
        print(cnt)
        print('fx', f_x)
        
        if cnt > 100:
            break
        cnt += 1

    print('- - - result - - - ')
    print(func_f(x))
    original_mat = G*A
    B, C = x_2_BC(x)
    recovered_mat = G*np.matmul(B,C)
    xs, ys = np.nonzero(G)
    print(np.stack([original_mat[xs,ys], recovered_mat[xs,ys]]).transpose())
    print(np.mean(np.abs(original_mat[xs,ys]-recovered_mat[xs,ys])))



