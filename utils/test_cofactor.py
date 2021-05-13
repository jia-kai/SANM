#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import itertools

def compute_cofactor(x):
    u, s, vt = np.linalg.svd(x)
    detsi = np.empty_like(s)
    for i in range(len(s)):
        detsi[i] = np.prod(s[:i]) * np.prod(s[i+1:])
    return np.linalg.det(u.dot(vt)) * u.dot(np.diag(detsi)).dot(vt)

def minor(arr, i, j):
    l0 = list(itertools.chain(range(i), range(i+1, arr.shape[0])))
    l1 = list(itertools.chain(range(j), range(j+1, arr.shape[1])))
    return arr[np.array(l0)[:,np.newaxis], np.array(l1)]

def compute_cofactor_brute(x):
    ret = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = np.linalg.det(minor(x, i, j))
            if (i + j) % 2:
                v = -v
            ret[i, j] = v
    return ret

def check(x):
    print('input')
    print(x)
    print('det:', np.linalg.det(x))
    c0 = compute_cofactor(x)
    c1 = compute_cofactor_brute(x)
    print('c0')
    print(c0)
    print('c1')
    print(c1)
    print('diff', np.abs(c0 - c1).max())

def main():
    x = np.random.uniform(1, 4, (5, 5))
    check(x)
    x[-1] = x[0]
    check(x)

if __name__ == '__main__':
    main()
