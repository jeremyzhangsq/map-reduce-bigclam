# NMF by alternative non-negative least squares using projected gradients
# This algorithm solves NMF by alternative non-negative least squares using projected gradients.
# It converges faster than the popular multiplicative update approach.
# Details and comparisons are in the following paper:
# C.-J. Lin. Projected gradient methods for non-negative matrix factorization. Neural Computation, 19(2007), 2756-2779.
# https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf

# We modified the original implementation to adapt to our own situation

from numpy import *
import numpy as np
from numpy.linalg import norm
from time import time
from sys import stdout
from numpy.linalg import lstsq


def NMF(G, k):
    """
    (W,H) = nmf(V,Winit,Hinit,tol,timelimit,maxiter)
    W,H: output solution
    Winit,Hinit: initial solution
    """
    tol = 0.00001  # tol: tolerance for a relative stopping condition
    timelimit = 10000  # timelimit: limit of time
    maxiter = 10000  # maxiter: limit of iterations
    V = G.matrix
    Winit = np.random.rand(np.size(V, 0), k)  # initial solution
    Hinit = np.random.rand(k, np.size(V, 1))  # initial solution

    W = Winit;
    H = Hinit;  # output solution
    initt = time();

    gradW = dot(W, dot(H, H.T)) - dot(V, H.T)
    gradH = dot(dot(W.T, W), H) - dot(W.T, V)
    initgrad = norm(r_[gradW, gradH.T])
    print('Init gradient norm %f' % initgrad)
    tolW = max(0.001, tol) * initgrad
    tolH = tolW

    for iter in range(1, maxiter):
        # stopping condition
        projnorm = norm(r_[gradW[logical_or(gradW < 0, W > 0)],
                           gradH[logical_or(gradH < 0, H > 0)]])
        if projnorm < tol * initgrad or time() - initt > timelimit: break

        (W, gradW, iterW) = nlssubprob(V.T, H.T, W.T, tolW, 1000)
        W = W.T  # W: output solution
        gradW = gradW.T

        if iterW == 1: tolW = 0.1 * tolW

        (H, gradH, iterH) = nlssubprob(V, W, H, tolH, 1000)  # H: output solution
        if iterH == 1: tolH = 0.1 * tolH

        if iter % 10 == 0: stdout.write('.')

    print('\nIter = %d Final proj-grad norm %f' % (iter, projnorm))

    C = {}
    for idx, row in enumerate(W):
        list_row = row.tolist()
        max_index = list_row.index(max(list_row))  # return index of max value
        C.setdefault(str(max_index), []).append(str(idx))

    return C


def nlssubprob(V, W, Hinit, tol, maxiter):
    """
    H, grad: output solution and gradient
    iter: #iterations used
    V, W: constant matrices
    Hinit: initial solution
    tol: stopping tolerance
    maxiter: limit of iterations
    """

    H = Hinit
    WtV = dot(W.T, V)
    WtW = dot(W.T, W)

    alpha = 0.05;
    beta = 0.3;
    for iter in range(1, maxiter):
        grad = dot(WtW, H) - WtV
        projgrad = norm(grad[logical_or(grad < 0, H > 0)])
        if projgrad < tol: break

        # search step size
        for inner_iter in range(1, 20):
            Hn = H - alpha * grad
            Hn = where(Hn > 0, Hn, 0)
            d = Hn - H
            gradd = sum(grad * d)
            dQd = sum(dot(WtW, d) * d)
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0;
            if inner_iter == 1:
                decr_alpha = not suff_decr;
                Hp = H;
            if decr_alpha:
                if suff_decr:
                    H = Hn;
                    break;
                else:
                    alpha = alpha * beta;
            else:
                if not suff_decr or (Hp == Hn).all():
                    H = Hp;
                    break;
                else:
                    alpha = alpha / beta;
                    Hp = Hn;

        if iter == maxiter:
            print('Max iter in nlssubprob')
    return (H, grad, iter)


# # Multiplicative Update Approach
# def mu(G, k):
#     '''
#     Run multiplicative updates to perform nonnegative matrix factorization on A.
#     Return matrices W, H such that A = WH.
#
#     Parameters:
#         A: ndarray
#             - m by n matrix to factorize
#         k: int
#             - integer specifying the column length of W / the row length of H
#             - the resulting matrices W, H will have sizes of m by k and k by n, respectively
#         delta: float
#             - float that will be added to the numerators of the update rules
#             - necessary to avoid division by zero problems
#         num_iter: int
#             - number of iterations for the multiplicative updates algorithm
#         init_W: ndarray
#             - m by k matrix for the initial W
#         init_H: ndarray
#             - k by n matrix for the initial H
#         print_enabled: boolean
#             - if ture, output print statements
#
#     Returns:
#         W: ndarray
#             - m by k matrix where k = dim
#         H: ndarray
#             - k by n matrix where k = dim
#     '''
#
#     A = G.matrix
#     delta = 0.0000001
#     num_iter = 5
#     init_W = None
#     init_H = None
#     print_enabled = True
#
#     print('Applying multiplicative updates on the input matrix...')
#
#     if print_enabled:
#         print('---------------------------------------------------------------------')
#         print('Frobenius norm ||A - WH||_F')
#         print('')
#
#     # Initialize W and H
#     if init_W is None:
#         W = np.random.rand(np.size(A, 0), k)
#     else:
#         W = init_W
#
#     if init_H is None:
#         H = np.random.rand(k, np.size(A, 1))
#     else:
#         H = init_H
#
#     # Decompose the input matrix
#     for n in range(num_iter):
#
#         # Update H
#         W_TA = W.T @ A
#         W_TWH = W.T @ W @ H + delta
#
#         for i in range(np.size(H, 0)):
#             for j in range(np.size(H, 1)):
#                 H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]
#
#         # Update W
#         AH_T = A @ H.T
#         WHH_T = W @ H @ H.T + delta
#
#         for i in range(np.size(W, 0)):
#             for j in range(np.size(W, 1)):
#                 W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]
#
#         if print_enabled:
#             frob_norm = np.linalg.norm(A - W @ H, 'fro')
#             print("iteration " + str(n + 1) + ": " + str(frob_norm))
#
#     C = {}
#     for idx, row in enumerate(W):
#         list_row = row.tolist()
#         max_index = list_row.index(max(list_row))  # return index of max value
#         C.setdefault(str(max_index), []).append(str(idx))
#
#     return C
#
#
# # Alternating Non-negative Least Squares
# def als(G, k):
#     '''
#     Run the alternating least squares method to perform nonnegative matrix factorization on A.
#     Return matrices W, H such that A = WH.
#
#     Parameters:
#         A: ndarray
#             - m by n matrix to factorize
#         k: int
#             - integer specifying the column length of W / the row length of H
#             - the resulting matrices W, H will have sizes of m by k and k by n, respectively
#         num_iter: int
#             - number of iterations for the multiplicative updates algorithm
#         print_enabled: boolean
#             - if ture, output print statements
#
#     Returns:
#         W: ndarray
#             - m by k matrix where k = dim
#         H: ndarray
#             - k by n matrix where k = dim
#     '''
#
#     A = G.matrix
#     num_iter = 10
#     init_W = None
#     init_H = None
#     print_enabled = True
#
#     print('Applying the alternating least squares method on the input matrix...')
#
#     if print_enabled:
#         print('---------------------------------------------------------------------')
#         print('Frobenius norm ||A - WH||_F')
#         print('')
#
#     # Initialize W and H
#     if init_W is None:
#         W = np.random.rand(np.size(A, 0), k)
#     else:
#         W = init_W
#
#     if init_H is None:
#         H = np.random.rand(k, np.size(A, 1))
#     else:
#         H = init_H
#
#     # Decompose the input matrix
#     for n in range(num_iter):
#         # Update H
#         # Solve the least squares problem: argmin_H ||WH - A||
#         H = lstsq(W, A, rcond=-1)[0]
#         # Set negative elements of H to 0
#         H[H < 0] = 0
#
#         # Update W
#         # Solve the least squares problem: argmin_W.T ||H.TW.T - A.T||
#         W = lstsq(H.T, A.T, rcond=-1)[0].T
#
#         # Set negative elements of W to 0
#         W[W < 0] = 0
#
#         if print_enabled:
#             frob_norm = np.linalg.norm(A - W @ H, 'fro')
#             print("iteration " + str(n + 1) + ": " + str(frob_norm))
#
#     C = {}
#     for idx, row in enumerate(W):
#         list_row = row.tolist()
#         max_index = list_row.index(max(list_row))  # return index of max value
#         C.setdefault(str(max_index), []).append(str(idx))
#
#     return C
