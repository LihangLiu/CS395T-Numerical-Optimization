import numpy as np 
import scipy.io

def init_X():
    return np.random.normal(0, 1, [n,m])

def get_A_b(mat):
    A, b = [], []
    for i in xrange(m):
        A.append(mat['A'][i][0])
        b.append(mat['b'][i][0].reshape([-1]))
    return np.array(A), np.array(b)

def norm2(vec):
    return np.matmul(vec, vec)

def norm1(vec):
    return np.sum(np.abs(vec))

def func_g(X):
    """
    return float
    """
    res = 0
    for i in xrange(m):
        res += norm2(np.matmul(A[i], X[:,i]) - b[i])
    return res

def func_gi(i, xi):
    """
    return float
    """
    return norm2(np.matmul(A[i], xi) - b[i])

def func_h(X):
    """
    return float
    """
    res = 0
    for i in xrange(m-1):
        for j in xrange(i+1,m):
            res += norm1(X[:,i] - X[:,j])
    return lamda*res

def func_diff_gi(i, xi):
    """
    return (n)
    """
    return 2*np.matmul(A[i].T, np.matmul(A[i], xi) - b[i])

def func_Gi(t, i, xi):
    """
    return (n)
    """
    res = 1.0/t*(xi - func_prox_th(t, i, xi - t*func_diff_gi(i, xi)))
    return res

def func_prox_th(t, i, xi_hat):
    """
    return (n)
    """
    # test
    # return np.array(xi_hat)

    ui = np.zeros_like(xi_hat)
    for k in xrange(n):
        xk = np.append(X[k,:i], X[k,i+1:])
        ui[k] = func_sub_prox_th(t, xk, xi_hat[k])
    return ui

def func_sub_prox_th(t, xk, xki):
    """
    xk: (m-1)
    xki: (1)
    return uki: (1)
    """
    uki = 0
    sorted_xk = np.array(np.sort(xk))
    m_ = sorted_xk.shape[0]
    patched_xk = np.append(-9999999, np.append(sorted_xk, 9999999))
    for j in xrange(0, patched_xk.shape[0]-1):
        begin, end = patched_xk[j], patched_xk[j+1]
        y = -m_ + 2*j
        uki = xki - lamda*y*2*t
        e0 = lamda*y*2*t + begin - xki
        e1 = lamda*y*2*t + end - xki
        if begin <= uki and uki <= end:
            break
        if e0 >= 0:
            uki = begin
            break
    return uki


if __name__ == '__main__':
    m, n = 10, 1000
    t_hat = 1.0
    beta = 0.5
    lamda = 1
    np.random.seed(0)
    mat_file = "hw4_data.mat"
    mat = scipy.io.loadmat(mat_file)
    A,b = get_A_b(mat)
    # print(A.shape, b.shape)

    X = init_X()
    cnt = 0
    if_proximal = False
    while True:
        for i in np.random.permutation(m):
            xi = X[:, i]

            t = t_hat
            while True:
                Gt_xi = func_Gi(t, i, xi)
                diff_g_xi = func_diff_gi(i, xi)
                if func_gi(i, xi-t*Gt_xi) <= func_gi(i, xi) - t*np.matmul(diff_g_xi, Gt_xi) + 0.5*t*norm2(Gt_xi):
                    break
                t = beta*t

            Gt_xi = func_Gi(t, i, xi)
            X[:, i] = xi - t*Gt_xi
            

        print('iter: %d, loss: %.2f' % (cnt, func_g(X) + func_h(X)))
        # print('iter', cnt, i)
        # print(t, func_g(X), func_g(X) + func_h(X))
        cnt += 1
        if cnt > 100:
            break

