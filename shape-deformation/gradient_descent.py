
import numpy as np
from scipy import linalg

from src.util import *

def init_all_xyz(n):
    xyzs = np.zeros([n,3])
    return xyzs

def xyzs_2_Rs(xyzs):
    Rs = np.zeros([n,3,3])
    for i in xrange(n):
        x,y,z = xyzs[i]
        M = np.array([[0,-z,y],
                        [z,0,-x],
                        [-y,x,0]])
        
        Rs[i] = linalg.expm(M)
        
    return Rs

def cal_p_diff(Rs, ps, ps_rest, hs, H_indice, lamda):
    ps_diff = np.zeros_like(ps)

    for ii, i in enumerate(H_indice):
        ps_diff[i] += 2*lamda*(ps[i] - hs[ii])

    for i in xrange(n):
        cNi = Ns[i]
        for j in cNi:
            ps_diff[i] += 2*(ps[i]-ps[j] - np.matmul(Rs[i], ps_rest[i]-ps_rest[j]))
            # test
            ps_diff[i] += 2*(np.matmul(Rs[j], ps_rest[j]-ps_rest[i]) - (ps[j]-ps[i]))

    return ps_diff

def cal_xyzs_diff(xyzs, Rs, ps, ps_rest):
    xyzs_diff = np.zeros_like(xyzs)

    L_over_Ri = np.zeros_like(Rs)   # (n,3,3)
    for i in xrange(n):
        cNi = Ns[i]
        for j in cNi:
            L_over_Ri[i] += 2*np.matmul(
                            (np.matmul(Rs[i], ps_rest[i]-ps_rest[j]) - (ps[i]-ps[j])).reshape([-1,1]), 
                            (ps_rest[i]-ps_rest[j]).reshape([-1,1]).transpose()
                        )

    Ri_over_Mi = np.array(Rs)        # (n,3,3)
    Mi_over_xi = np.stack([np.array([[0,0,0],   # (n,3,3)
                                    [0,0,-1],
                                    [0,1,0]])]*n)
    Mi_over_yi = np.stack([np.array([[0,0,1],   # (n,3,3)
                                    [0,0,0],
                                    [-1,0,0]])]*n)
    Mi_over_zi = np.stack([np.array([[0,-1,0],   # (n,3,3)
                                    [1,0,0],
                                    [0,0,0]])]*n)

    xyzs_diff[:,0] = np.mean(L_over_Ri*Ri_over_Mi*Mi_over_xi, axis=(1,2))
    xyzs_diff[:,1] = np.mean(L_over_Ri*Ri_over_Mi*Mi_over_yi, axis=(1,2))
    xyzs_diff[:,2] = np.mean(L_over_Ri*Ri_over_Mi*Mi_over_zi, axis=(1,2))

    return xyzs_diff

if __name__ == '__main__':
    epsilon = 0.1
    alpha = 0.01
    lamda = 1
    n = 50*50
    fig_title = 'Gradient Descent'
    ps_rest, Ns = init_all_points(n)    # (n, 3); map
    hs, H_indice = init_all_h(n)      # (5, 3); (5)
    xyzs = init_all_xyz(n)          # (n, 3)
    # print 'Rs'
    # print xyzs_2_Rs(xyzs)
    # visualize_p(ps_rest, hs, H_indice, int(np.sqrt(n)), title=fig_title)

    ps = np.array(ps_rest)
    iteri = 1
    while True:
        print 'iter', iteri
        Rs = xyzs_2_Rs(xyzs)
        ps_diff = cal_p_diff(Rs, ps, ps_rest, hs, H_indice, lamda)
        xyzs_diff = cal_xyzs_diff(xyzs, Rs, ps, ps_rest)

        ps = ps - alpha*ps_diff
        xyzs = xyzs - alpha*xyzs_diff

        if iteri % 2000 == 0:
            # break
            visualize_p(ps, hs, H_indice, int(np.sqrt(n)), title=fig_title)
            
        iteri += 1

    visualize_p(ps, hs, H_indice, int(np.sqrt(n)), title=fig_title)








