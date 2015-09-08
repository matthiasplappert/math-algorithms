from argparse import ArgumentParser
import timeit

import numpy as np


def lr_decomp(A):
    if A.ndim != 2:
        raise ValueError('A must have 2 dimensions')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix')
    A = np.copy(A)

    n = A.shape[0]
    for i in xrange(n - 1):
        for j in xrange(i + 1, n):
            c = -A[j, i] / A[i, i]
            for k in xrange(i, n):
                A[j, k] += A[i, k] * c
            A[j, i] = -c

    R = np.triu(A)
    L = np.tril(A)
    np.fill_diagonal(L, 1.)
    return L, R


def cholesky_decomp(A):
    if A.ndim != 2:
        raise ValueError('A must have 2 dimensions')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix')

    n = A.shape[0]
    L = np.zeros(A.shape)
    for i in xrange(n):
        # diagonal elements
        s = 0
        for j in xrange(i):
            s += np.square(L[i, j])
        sqrt = np.sqrt(A[i, i] - s)
        if np.isnan(sqrt):
            raise ValueError('A is not a symmetric positive-definite matrix')
        L[i, i] = sqrt

        # elements in the rows below the current element (i,i)
        for k in xrange(i + 1, n):
            s = 0
            for j in xrange(i):
                s += L[k, j] * L[i, j]
            L[k, i] = (A[k, i] - s) / L[i, i]
    return L, L.T


def solve(L, R, b):
    if L.shape != R.shape:
        raise ValueError('L and R must have equal shapes')
    if L.ndim != 2:
        raise ValueError("L must have 2 dimensions")
    if L.shape[0] != L.shape[1]:
        raise ValueError('L must be a square matrix')
    if b.ndim != 1:
        raise ValueError('b must be a vector')
    if b.shape[0] != L.shape[0]:
        raise ValueError('b must fit the dimension of L')
    n = L.shape[0]

    # Solve Ax = b where A = LR. Solve Ly = b first ...
    y = np.zeros(b.shape)
    y[0] = b[0] / L[0, 0]
    for i in xrange(1, n):
        s = 0
        for j in xrange(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]

    # ... and then Rx = y.
    x = np.zeros(b.shape)
    x[n - 1] = y[n - 1] / R[n - 1, n - 1]
    for i in xrange(n - 2, -1, -1):
        s = 0
        for j in xrange(n - 1, i, -1):
            s += R[i, j] * x[j]
        x[i] = (y[i] - s) / R[i, i]
    return x


def main(args):
    n_elems = args.n_elems

    # Create random matrix A and matching vector b since we want to solve Ax = b.
    A = np.random.random((n_elems, n_elems))
    A = np.dot(A, A.T)  # make it s.p.d.
    b = np.random.random(n_elems)

    print('solving with LR decomposition ...')
    start = timeit.default_timer()
    L, R = lr_decomp(A)
    x = solve(L, R, b)
    duration = timeit.default_timer() - start
    print('correct result' if np.allclose(np.dot(A, x), b) else 'wrong result')
    print('done, took %fs' % duration)
    print('')

    print('solving with Cholesky decomposition ...')
    start = timeit.default_timer()
    L, R = cholesky_decomp(A)
    x = solve(L, R, b)
    duration = timeit.default_timer() - start
    print('correct result' if np.allclose(np.dot(A, x), b) else 'wrong result')
    print('done, took %fs' % duration)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n-elems', type=int, default=100)
    main(parser.parse_args())
