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

def func_h(X):
    """
    return float
    """
    res = 0
    for i in xrange(m-1):
        for j in xrange(i+1,m):
            res += norm1(X[:,i] - X[:,j])
    return lamda*res

def func_diff_g(X):
    """
    return (m,n)
    """
    diff_g = np.zeros_like(X)
    for i in xrange(m):
        diff_g[:, i] = 2*np.matmul(A[i].T, np.matmul(A[i], X[:,i]) - b[i])
    return diff_g

def func_G(t, X):
    """
    return (m,n)
    """
    return 1.0/t*(X - func_prox(t, X - t*func_diff_g(X)))

def func_prox(t, X):
    """
    return (m,n)
    """
    prox_X = np.zeros_like(X)
    for i in xrange(n):
        prox_X[i] = func_sub_prox(t, X[i])
    return prox_X

def tmp_func_f(u, x):
    res = 0.0
    for k in xrange(m):
        for l in xrange(m):
            res += lamda*np.abs(u[k] - u[l])
        res += 0.5*(u[k]-x[k])
    return res

def func_sub_prox(t, x):
    """
    x: (1,m)
    """
    # print('before', tmp_func_f(x,x))
    index = np.argsort(x)
    sorted_x = np.array(x[index])
    # sorted x
    u = np.array(sorted_x)
    # print(u)
    for _ in xrange(1):
        for i in xrange(m):
            ui = x[i] - t*lamda*(2*i-m+1)
            if i == 0:
                ui = np.clip(ui, -9999999, u[i+1])
            elif i == m-1:
                ui = np.clip(ui, u[i-1], 9999999)
            else:
                ui = np.clip(ui, u[i-1], u[i+1])
            u[i] = ui
        # print('---------------')
        # print('ui', tmp_func_f(u, sorted_x))
        # print(u)
        u += (np.mean(sorted_x) - np.mean(u))
        # print('average ui', tmp_func_f(u, sorted_x))
        # print(u)

        # for i in xrange(m-1, -1, -1):
        #     ui = x[i] - 2*lamda*(2*i-m+1)
        #     if i == 0:
        #         ui = np.clip(ui, -9999999, u[i+1])
        #     elif i == m-1:
        #         ui = np.clip(ui, u[i-1], 9999999)
        #     else:
        #         ui = np.clip(ui, u[i-1], u[i+1])
        #     u[i] = ui
        # # print('ui', tmp_func_f(u, sorted_x))
        # # print(u)
        # u += (np.mean(sorted_x) - np.mean(u))
        # # print('average ui', tmp_func_f(u, sorted_x))
        # # print(u)

    # exit(0)
    # resume unsorted
    unsorted_u = np.zeros_like(u)
    unsorted_u[index] = u
    # print(x)
    # print(unsorted_u)
    # exit(0)
    # print('after', tmp_func_f(unsorted_u,x))
    # exit(0)
    return unsorted_u


if __name__ == '__main__':
    m, n = 10, 1000
    t_hat = 1.0
    beta = 0.5
    lamda = 1.0
    np.random.seed(0)
    mat_file = "hw4_data.mat"
    mat = scipy.io.loadmat(mat_file)
    A,b = get_A_b(mat)
    print(A.shape, b.shape)

    X = init_X()
    cnt = 0
    while True:
        t = t_hat
        while True:
            Gt_X = func_G(t, X)
            diff_g_X = func_diff_g(X)
            if func_g(X-t*Gt_X) <= func_g(X) - t*np.matmul(diff_g_X.reshape([-1]), Gt_X.reshape([-1])) + 0.5*t*norm2(Gt_X.reshape([-1])):
                break
            t = beta*t

        # test
        # t = 0.001

        Gt_X = func_G(t, X)
        X = X - t*Gt_X
        # test
        # X = X - t*func_diff_g(X)

        print('iter', cnt)
        print(t, func_g(X), func_g(X) + func_h(X))
        cnt += 1

