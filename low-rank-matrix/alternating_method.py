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
    # return np.zeros([N*R*2])

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

def func_f(x):
    res = 0
    B, C = x_2_BC(x)
    D = np.matmul(B, C)
    for i in xrange(N):
        for j in xrange(N):
            res += G[i,j]*(A[i,j]-D[i,j])**2
    res += 0.5*mu*(np.sum(np.square(B)) + np.sum(np.square(C)))
    return res

def solve_B():
    M = np.zeros([N, R, N, R])
    Y = np.zeros([N, R])
    for k in xrange(N):
        for l in xrange(R):
            # collect diff_f/diff_bkl
            M[k,l,k,l] += mu
            for i in xrange(N):
                for j in xrange(N):
                    g_2_clj = G[i,j]*2*(func_diff_d_b(i,j,k,l))
                    for o in xrange(R):
                        M[k,l,i,o] += g_2_clj*C[o,j]
                    Y[k,l] += g_2_clj*A[i,j]

    X = np.linalg.solve(M.reshape([N*R,N*R]), Y.reshape([N*R]))
    return X.reshape([N,R])

def solve_C():
    M = np.zeros([R, N, R, N])
    Y = np.zeros([R, N])
    for m in xrange(R):
        for n in xrange(N):
            # collect diff_f/diff_bkl
            M[m,n,m,n] += mu
            for i in xrange(N):
                for j in xrange(N):
                    g_2_bim = G[i,j]*2*(func_diff_d_c(i,j,m,n))
                    for o in xrange(R):
                        M[m,n,o,j] += g_2_bim*B[i,o]
                    Y[m,n] += g_2_bim*A[i,j]

    X = np.linalg.solve(M.reshape([R*N,R*N]), Y.reshape([R*N]))
    return X.reshape([R,N])


if __name__ == '__main__':
    N = 10
    R = 2
    mu = 0.1
    np.random.seed(21)
    A = init_A()
    G = init_G()
    delta_hat = 1
    delta = 0.5
    eta = 0.2
    x = init_x()
    cnt = 0
    while True:
        B, C = x_2_BC(x)

        B = solve_B()
        C = solve_C()

        x = BC_2_x(B, C)
        print('iteration:', cnt, 'loss:', func_f(x))

        if cnt > 30:
            break
        cnt += 1

    print('- - - result - - - ')
    original_mat = G*A
    B, C = x_2_BC(x)
    recovered_mat = G*np.matmul(B,C)
    xs, ys = np.nonzero(G)
    print("original | recovered")
    print(np.stack([original_mat[xs,ys], recovered_mat[xs,ys]]).transpose())



