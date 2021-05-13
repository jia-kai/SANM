#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def main():
    coeffs = [0,1.40105e-05,-0.00298675,-0.00212753,-0.0044604,-0.0223296,-0.00765384,0.00529307,0.00500442]
    bnd = 0.1
    vx = np.linspace(-bnd, bnd, 100)
    vy = np.polyval(coeffs[::-1], vx)
    plt.plot(vx, vy)
    plt.show()

if __name__ == '__main__':
    main()
