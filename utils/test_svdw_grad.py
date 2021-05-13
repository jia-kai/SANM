#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools

rng = np.random.RandomState(42)

def svdw(m):
    n = m.shape[0]
    assert m.shape == (n, n)
    u, s, vt = np.linalg.svd(m)
    w = u @ vt
    assert np.allclose(u.T @ u, np.eye(n))
    assert np.allclose(w.T @ w, np.eye(n))
    assert np.allclose(u @ np.diag(s) @ u.T @ w, m)
    return u, s, w

def check_eq(msg, a, b):
    diff = np.abs(a - b).max()
    assert diff < 1e-5, (msg, diff)

def svdw_jacobian(M, u, s, w):
    n = M.shape[0]
    assert M.shape == u.shape == w.shape == (n, n)
    v = w.T @ u
    dsdm = np.empty((n, n*n), dtype=M.dtype)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dsdm[i, j*n+k] = u.T[i, j] * v[k, i]

    dwdy = np.empty((n*n, n*n), dtype=M.dtype)
    dydm = np.empty_like(dwdy)
    dudx = np.empty_like(dwdy)
    dxdm = np.empty_like(dwdy)
    for i, j, k, l in itertools.product(range(n), range(n), range(n), range(n)):
        cij = u.T[i, k] * v[l, j]
        cji = u.T[j, k] * v[l, i]
        dydm[i*n+j, k*n+l] = 0 if i == j else (cij - cji) / (s[i] + s[j])
        dwdy[i*n+j, k*n+l] = u[i, k] * v.T[l, j]
        dudx[i*n+j, k*n+l] = 0 if l != j else u[i, k]
        dxdm[i*n+j, k*n+l] = 0 if i == j else (
            cij * s[j] + cji * s[i]) / (s[j]**2 - s[i]**2)

    return dudx @ dxdm, dsdm, dwdy @ dydm

def svdw_jacobian_num(M, u, s, w, eps=1e-4):
    n = M.shape[0]
    assert M.shape == (n, n)
    dudm = np.zeros((n*n, n*n), dtype=M.dtype)
    dsdm = np.zeros((n, n*n), dtype=M.dtype)
    dwdm = np.zeros((n*n, n*n), dtype=M.dtype)
    grad = lambda x, y: (y - x).flatten() / (eps * 2)
    for i in range(n):
        for j in range(n):
            x0 = M[i, j]
            M[i, j] = x0 - eps
            u1, s1, w1 = svdw(M)
            M[i, j] = x0 + eps
            u2, s2, w2 = svdw(M)
            M[i, j] = x0
            p = i*n+j
            dudm[:, p] = grad(u1, u2)
            dsdm[:, p] = grad(s1, s2)
            dwdm[:, p] = grad(w1, w2)

    return dudm, dsdm, dwdm

def main():
    np.set_printoptions(4, suppress=True)
    n = 8
    m = rng.normal(size=(n, n))
    u, s, w = svdw(m)
    print('det(m):', np.linalg.det(m))
    print('s:', s)
    for gsym, gnum, name in zip(svdw_jacobian(m, u, s, w),
                                svdw_jacobian_num(m, u, s, w),
                                ['dU/dM', 'dS/dM', 'dW/dM']):
        print(f'====== {name}')
        if gsym.shape[0] == gsym.shape[1]:
            print('grad det:', np.linalg.det(gsym))
            print('grad rank:', np.linalg.matrix_rank(gsym))
        diff = np.abs(gsym - gnum).mean()
        print(diff, diff / np.abs(gnum).mean())

if __name__ == '__main__':
    main()
