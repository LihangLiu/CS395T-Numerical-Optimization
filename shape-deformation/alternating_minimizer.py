
import numpy as np

from src.util import *


def solve_R(Rs, ps, ps_rest):
    new_Rs = np.array(Rs)

    for i in xrange(n):
        cNi = Ns[i]
        P = np.array([ps_rest[i]-ps_rest[j] for j in cNi])  # (*,3)
        Q = np.array([ps[i]-ps[j] for j in cNi])            # (*,3)
        S = np.matmul(P.transpose(), Q)
        U, s, V = np.linalg.svd(S)
        Ri = np.matmul(V, U.transpose())
        new_Rs[i] = Ri

    return new_Rs

def solve_p(Rs, ps, ps_rest, hs, H_indice, lamda):
    """
    x,y,z axis are taken independently
    """
    new_ps = np.array(ps)   # (n, 3)

    for axis in xrange(3):
        B = np.zeros([n,n])     # (n,n)
        Y = np.zeros([n])     # (n,n)
        for i in xrange(n):
            cNi = Ns[i]
            # for Ri
            for j in cNi:
                B[i, i] += 2
                B[i, j] += (-2)
                Y[i] += (np.matmul(Rs[i], ps_rest[i]-ps_rest[j]) - np.matmul(Rs[j], ps_rest[j]-ps_rest[i]))[axis]

        # for h
        for ii, i in enumerate(H_indice):
            B[i, i] += lamda
            Y[i] += lamda*hs[ii][axis]

        new_ps[:, axis] = np.linalg.solve(B, Y)

    return new_ps


if __name__ == '__main__':
    epsilon = 0.1
    lamda = 0.5
    n = 100*100
    fig_title = 'Alternating Minimizer'
    ps_rest, Ns = init_all_points(n)    # (n, 3); map
    hs, H_indice = init_all_h(n)      # (5, 3); (5)
    Rs = init_all_R(n)      # (n, 3, 3)
    visualize_p(ps_rest, hs, H_indice, int(np.sqrt(n)), title=fig_title)

    ps = np.array(ps_rest)
    cnt = 0
    while True:
        print 'iter ...'
        Rs = solve_R(Rs, ps, ps_rest)
        ps = solve_p(Rs, ps, ps_rest, hs, H_indice, lamda)
        #if points_diff(old_ps, ps) < epsilon:
        #    break
        if cnt > 8:
            break
        cnt += 1
        old_ps = np.array(ps)

    visualize_p(ps, hs, H_indice, int(np.sqrt(n)), title=fig_title)







