import numpy as np
import random
import operator

def log_likelihood(F, A):
    """implements equation 2 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    """
    A_soft = F.dot(F.T)
    # Next two lines are multiplied with the adjacency matrix, A
    # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
    FIRST_PART = A * np.log(1. - np.exp(-1. * A_soft))
    sum_edges = np.sum(FIRST_PART)
    SECOND_PART = (1 - A) * A_soft
    sum_nedges = np.sum(SECOND_PART)
    log_likeli = sum_edges - sum_nedges
    return log_likeli


def sigm(x):
    """auxiliary function for gradient:
    exp(x)/(1-exp(x)), where x = F_u \dot F_v^T
    """
    return np.divide(np.exp(-1. * x), 1. - np.exp(-1. * x))


def gradient(F, adjlst, i, sum_nneigh):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
      * i indicates the row under consideration
    """
    N, C = F.shape

    neighbours = adjlst[i]

    sum_nneigh -= F[i]
    sum_neigh = np.zeros((C,))
    for nb in neighbours:
        dotproduct = F[nb].dot(F[i])
        sum_neigh += F[nb] * sigm(dotproduct)
        sum_nneigh -= F[nb] # speed up non neighbor computation in eq.4

    grad = sum_neigh - sum_nneigh
    return grad

def getConductance(adjlst, vset, m):
    cut = 0
    vol = 0
    for v in vset:
        for nghr in adjlst[v]:
            vol += 1
            if nghr not in vset:
                cut += 1
    return cut/float(min(vol,m-vol))

def localMinNeig(G):
    maps = {}
    adjlst = G.list
    m = G.m
    for v in adjlst:
        vset = [e for e in adjlst[v]]
        vset.append(v)
        maps[v] = getConductance(adjlst,vset,m)
    return sorted(maps.items(),key=operator.itemgetter(1))

def rndInit(G, F, k, vertex):
    while len(vertex):
        for v in vertex:
            vset = [e for e in G.list[v]]
            vset.append(v)
            i = random.randrange(k)
            for ele in vset:
                if ele in vertex:
                    vertex.remove(ele)
                    F[ele,i] += 1
            break

def commInit(G, k, epsilon):
    F = np.full((G.n, k), epsilon)
    lists = localMinNeig(G)
    vertexs = set(G.vertex)
    cnt = 0
    for vt, val in lists:
        if vt not in vertexs:
            continue
        vset = G.list[vt]
        vset.append(vt)
        if cnt == k:
            break
        for v in vset:
            if v in vertexs:
                vertexs.remove(v)
            F[v,cnt] = 1
        cnt += 1


    # random assign some user for no-member community
    while cnt < k:
        for i in range(10):
            v = random.sample(G,vertexs,1)
            F[v, cnt] = 1
        cnt += 1

    return F


def bigClam(G, k, theshold=0.00001):
    yita = 0.0005  # todo: tunable parameter for gradient update
    epsilon = 10 ** (-8)  # background edge propability in sec. 4
    delta = np.sqrt(-np.log(1 - epsilon))  # threshold to determine user-community edge
    graph = G.matrix
    adjlst = G.list
    N = G.n
    # change F init to local minimal neighborhood
    # src: https://snap.stanford.edu/snap/doc/snapuser-ref/dd/d81/classTCoda.html#a132e9f32c4ad4329d70dd555fc7b8cf0
    F = commInit(G, k, epsilon)

    # change F init to random init
    # F = rndInit(G, F, k, set(G.vertex))

    ll = np.infty
    while True:
        # todo: check the correctness of pre store in eq.4
        sum_nneigh = np.sum(F, axis=0)
        for person in range(N):
            grad = gradient(F, adjlst, person, sum_nneigh)
            F[person] += yita * grad
            F[person] = np.maximum(epsilon, F[person])  # F should be nonnegative
        newll = log_likelihood(F, graph)
        dt = abs(ll - newll)
        print('At step %5i %5.3f ll is %5.3f' % (n, dt, ll))
        if dt < theshold:
            break
        ll = newll

    C = []
    for j in range(k):
        c = []
        for i in range(N):
            if F[i, j] >= delta:
                c.append(i)
        C.append(c)
    return C
