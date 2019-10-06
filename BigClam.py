import numpy as np



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


def gradient(F, adjlst, i):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
      * i indicates the row under consideration
    """
    N, C = F.shape

    neighbours = adjlst[i]
    sum_nneigh = np.sum(F, axis=0) # pre store in eq.4
    sum_nneigh -= F[i]
    sum_neigh = np.zeros((C,))
    for nb in neighbours:
        dotproduct = F[nb].dot(F[i])
        sum_neigh += F[nb] * sigm(dotproduct)
        sum_nneigh -= F[nb] # speed up non neighbor computation in eq.4

    grad = sum_neigh - sum_nneigh
    return grad


def bigClam(graph, adjlst, k, theshold=0.00001):
    yita = 0.005  # todo: tunable parameter for gradient update
    epsilon = 10 ** (-8)  # background edge propability in sec. 4
    delta = np.sqrt(-np.log(1 - epsilon))  # threshold to determine user-community edge
    N = graph.shape[0]
    """todo: change F init to local minimal neighborhood 
    src: https://snap.stanford.edu/snap/doc/snapuser-ref/dd/d81/classTCoda.html#a132e9f32c4ad4329d70dd555fc7b8cf0
    """
    F = np.random.rand(N, k)
    ll = np.infty
    while True:
        for person in range(N):
            grad = gradient(F, adjlst, person)
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
