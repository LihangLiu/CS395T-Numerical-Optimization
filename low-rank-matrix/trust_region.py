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

def func_diff_f_b_b(k1, l1, k2, l2):
    res = 0
    D = np.matmul(B, C)
    for i in xrange(N):
        for j in xrange(N):
            res += G[i,j]*2 * func_diff_d_b(i,j,k1,l1) * func_diff_d_b(i,j,k2,l2)
            # res += G[i,j]*2*(A[i,j]-D[i,j]) * (-func_diff_d_b_b(i,j,k1,l1,k2,l2))     ## would be 0
    if k1==k2 and l1==l2:
        res += mu

    return res

def func_diff_f_b_c(k, l, m, n):
    res = 0
    D = np.matmul(B, C)
    for i in xrange(N):
        for j in xrange(N):
            res += G[i,j]*2 * func_diff_d_c(i,j,m,n) * func_diff_d_b(i,j,k,l)
            # res += G[i,j]*2*(A[i,j]-D[i,j]) * (-func_diff_d_b_c(i,j,k,l,m,n))         ## move to the line below
    if m == l:
        res += G[k,n]*2*(A[k,n]-D[k,n])*(-1)

    return res

def func_diff_f_c_c(m1, n1, m2, n2):
    res = 0
    D = np.matmul(B, C)
    for i in xrange(N):
        for j in xrange(N):
            res += G[i,j]*2 * func_diff_d_c(i,j,m1,n1) * func_diff_d_c(i,j,m2,n2)
            # res += G[i,j]*2*(A[i,j]-D[i,j]) * (-func_diff_d_c_c(i,j,m1,n1,m2,n2))     ## would be 0
    if m1==m2 and n1==n2:
        res += mu

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

def func_H():
    H = np.zeros([N*R*2, N*R*2])
    for k1 in xrange(N):
        for l1 in xrange(R):
            for k2 in xrange(N):
                for l2 in xrange(R):
                    cdiff = func_diff_f_b_b(k1, l1, k2, l2)
                    H[k1*R+l1, k2*R+l2] = cdiff
                    H[k2*R+l2, k1*R+l1] = cdiff
            
            for m in xrange(R):
                for n in xrange(N):
                    cdiff = func_diff_f_b_c(k1, l1, m, n)
                    H[k1*R+l1, N*R + m*N+n] = cdiff
                    H[N*R + m*N+n, k1*R+l1] = cdiff

    for m1 in xrange(R):
        for n1 in xrange(N):
            for m2 in xrange(R):
                for n2 in xrange(N):
                    cdiff = func_diff_f_c_c(m1, n1, m2, n2)
                    H[N*R + m1*N+n1, N*R + m2*N+n2] = cdiff
                    H[N*R + m2*N+n2, N*R + m1*N+n1] = cdiff

    return H

def func_m(f, g, H, p):
    # gTp = np.matmul(g.reshape([1,-1]), p)
    gTp = np.matmul(g, p)
    pTHp = np.matmul(p, np.matmul(H, p))
    # print('-')
    # print(gTp)
    # print(pTHp)
    return f + gTp + 0.5*pTHp

###################
# trust region subproblem
###################

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def Cauchy_point_calculation(g, H):
    p = -delta*g/norm(g)
    gTBg = np.matmul(g, np.matmul(H, g))
    if gTBg <= 0:
        tau = 1
    else:
        tau = min(1, np.power(norm(g), 3) / (delta*gTBg))
    return tau*p

# def exact_argmin_p_m(g, H):
#     w, v = linalg.eig(H)
#     print(w)
#     print(v)
#     index_n = np.argmin(w)
#     print(w[index_n])
#     print(v[:, index_n])
#     if np.matmul(v[:,index_n], g) == 0:
#         return 
#     exit(0)

def argmin_p_m(g, H):
    return Cauchy_point_calculation(g, H)


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
        g = func_g()
        H = func_H()
        p = argmin_p_m(g, H)

        f_x = func_f(x)
        f_xp = func_f(x+p)
        m0 = func_m(f_x, g, H, np.zeros(p.shape))
        mp = func_m(f_x, g, H, p)
        rho = (f_x - f_xp) / (m0 - mp + 0.000001)

        # print('---------------')
        print('iteration:', cnt, 'loss:', f_x)
        # print(cnt)
        # print(G)
        # print(A)
        # print(B)
        # print(C)
        # print(np.matmul(B, C))
        # print(g)
        # print(H)
        # print(p)
        # print(np.sum(np.square(p)))
        # print('fx', f_x)
        # print('fxp', f_xp)
        # print('m0', m0)
        # print('mp', mp)
        # print('mp/2', func_m(f_x, g, H, p*0.5))
        # print('rho', rho)
        # print('delta', delta)
        # print('p')
        # print(np.sum(np.abs(p)))
        if cnt > 100:
            break
        cnt += 1

        if rho < 0.25:
            new_delta = 0.25*delta
        else:
            if rho > 0.75 and norm(p) == delta:
                new_delta = min(2*delta, delta_hat)
            else:
                new_delta = delta

        if rho >= eta:
            new_x = x + p
        else:
            new_x = x

        delta = new_delta
        x = new_x

    print('- - - result - - - ')
    # print(func_f(x))
    original_mat = G*A
    B, C = x_2_BC(x)
    recovered_mat = G*np.matmul(B,C)
    xs, ys = np.nonzero(G)
    print("original | recovered")
    print(np.stack([original_mat[xs,ys], recovered_mat[xs,ys]]).transpose())
    # print(np.mean(np.abs(original_mat[xs,ys]-recovered_mat[xs,ys])))


