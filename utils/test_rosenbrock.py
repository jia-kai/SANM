#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize, line_search
import matplotlib.pyplot as plt

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def test_gd():
    xi = np.array([1.3, 0.7, 0.8, 1.9, 1.2], dtype=np.float64)
    # value found by one iteration of ANO (leading to local minimum)
    # xi[:] = [1.09666,0.8126,0.934773,1.07724,1.39017]
    mom = np.zeros_like(xi)
    lr = np.float64(0.0007)
    for i in range(20):
        gx = -rosen_der(xi)
        # mom = mom * 0.9 + gx
        # xi -= lr * mom
        # actually converges faster without line search
        lr, *_ = line_search(rosen, rosen_der, xi, gx)
        print(i, rosen(xi), lr)
        xi += lr * gx

def test_gx0():
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    x0 = np.array([0.55210179903703016, 0.20098467487614619,
                   -0.086018900221027095, 0.38088062880673074,
                   0.042037353198560869])
    print('x0', x0)
    gx0 = rosen_der(x0)
    print('gx0', gx0)
    px = np.linspace(-0.002, 0.002, 100)
    py = [rosen(x0 + gx0 * i) for i in px]

    plt.plot(px, py)
    plt.show()

def plot_poly():
    coeff = np.array([
        0.0244193, 0.0451206, 23.2877, 0.760964, 0.0265465, 0, 0, 0, 0,
    ])
    def ev(x):
        return np.power(x, np.arange(coeff.size)).dot(coeff)

    # px = np.linspace(-0.051765, 0.051765, 100)
    px = np.linspace(-1e-2, 1e-2, 100)
    py = [ev(i) for i in px]
    plt.plot(px, py)
    plt.show()

if __name__ == '__main__':
    # test_gd()
    test_gx0()
    # plot_poly()
