from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def init_all_points(n):
    ps = np.zeros([n, 3], dtype=np.float64)
    side_len = int(np.sqrt(n))
    for i in xrange(side_len):
        for j in xrange(side_len):
            ps[i+j*side_len] = [i,j,0]

    Ns = {}
    for i in xrange(side_len-1):
        for j in xrange(side_len-1):
            n0 = i+j*side_len
            n1 = i+j*side_len+1
            n2 = i+(j+1)*side_len
            n3 = i+(j+1)*side_len+1
            for ni in [n0,n1,n2,n3]:
                if not ni in Ns:
                    Ns[ni] = []

            Ns[n0] += [n1,n2,n3]
            for ni in [n1,n2,n3]:
                Ns[ni] += [n0]

            ## test
            # Ns[n0] += [n1, n2]
            # Ns[n1] += [n0, n3]
            # Ns[n2] += [n0, n3]
            

    for i in xrange(side_len-1):
        n0 = i+(side_len-1)*side_len
        n1 = i+(side_len-1)*side_len+1
        Ns[n0] += [n1]
        Ns[n1] += [n0]
    for j in xrange(side_len-1):
        n0 = side_len-1+j*side_len
        n1 = side_len-1+(j+1)*side_len
        Ns[n0] += [n1]
        Ns[n1] += [n0]

    # print Ns
    return ps, Ns

def init_all_R(n):
    res = np.stack([np.eye(3)]*n)
    return res

def init_all_h(n):
    side_len = int(np.sqrt(n))
    H_indice = [0, 
                side_len-1, 
                side_len*(side_len-1), 
                side_len*side_len-1, 
                side_len*(side_len/2)+side_len/2]
    hs = np.array([[0,0,0],
                    [side_len-1,0,0],
                    [0,side_len-1,0],
                    [side_len-1,side_len-1,0],
                    [side_len*0.5,side_len*0.5,side_len*0.1]])
    return hs, H_indice

def visualize_p(points, hs, H_indice, N, title=""):
    print points.shape

    # draw 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    dim = N
    # ax.set_xlim(0, dim)
    # ax.set_ylim(0, dim)
    # ax.set_zlim(0, dim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # points
    xs,ys,zs = points[:,0],points[:,1],points[:,2]
    rgbs = np.full((points.shape[0],3), 0)
    rgbs[:, 0] = 1
    ax.scatter(xs,ys, zs, color=rgbs) #, s=5)
    
    # hs
    xs,ys,zs = hs[:,0],hs[:,1],hs[:,2]
    rgbs = np.full((hs.shape[0],3), 0)
    rgbs[:, 1] = 1
    # ax.scatter(xs,ys, zs, color=rgbs) #, s=5)
    ax.scatter(xs[:4],ys[:4], zs[:4], color=rgbs) #, s=5)

    # h indice
    xs,ys,zs = points[H_indice,0],points[H_indice,1],points[H_indice,2]
    rgbs = np.full((xs.shape[0],3), 0)
    rgbs[:, 2] = 1
    ax.scatter(xs, ys, zs, color=rgbs) #, s=5)

    plt.show()

def points_diff(ps1, ps2):
    return np.mean(np.square(ps1-ps2))










