#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

mu = 3448275.8620689656
k = 33333333.33333334

def make_shape_matrix(shape):
    assert shape.shape == (4, 3)
    ret = np.empty((3, 3), dtype=shape.dtype)
    for i in range(3):
        ret[:, i] = shape[i + 1] - shape[0]
    return ret

def comptute_deformation_gradient(
        shape_rest: np.ndarray, shape_deform: np.ndarray):
    assert shape_rest.shape == shape_deform.shape == (4, 3)

    Dm = make_shape_matrix(shape_rest)
    Ds = make_shape_matrix(shape_deform)
    return Ds @ np.linalg.inv(Dm)

def compute_cauchy_stress(F: np.ndarray):
    b = F @ F.T
    J = np.linalg.det(F)
    Ic = np.trace(b)

    J53 = J ** (-5. / 3.)
    eye = np.eye(3)
    return mu * J53 * b - mu / 3 * J53 * Ic * eye + k * (J - 1) * eye

def compute_vtx_norm(shape_mat: np.ndarray):
    ret = np.empty((3, 4), dtype=shape_mat.dtype)
    ret[:, 1:] = -np.abs(np.linalg.det(shape_mat)) * np.linalg.inv(shape_mat).T
    ret[:, 0] = -ret[:, 1:].sum(axis=1)
    ret /= 6
    assert np.allclose(ret, compute_vtx_norm_chk(shape_mat))
    return ret

def compute_vtx_norm_chk(shape_mat: np.ndarray):
    v0, v1, v2 = shape_mat.T
    ret = np.empty((3, 4), dtype=shape_mat.dtype)
    ret[:, 1] = np.cross(v1, v2)
    ret[:, 2] = np.cross(v2, v0)
    ret[:, 3] = np.cross(v0, v1)
    if v0.dot(np.cross(v1, v2)) > 0:
        ret[:, 1:] = -ret[:, 1:]
    ret[:, 0] = -ret[:, 1:].sum(axis=1)
    return ret / 6

def main():
    shape_deform = np.zeros((4, 3), dtype=np.float64)
    angle = np.pi * 2 / 3
    spacing = 0.025
    shape_deform[:3, 0] = spacing * np.cos(np.arange(3) * angle)
    shape_deform[:3, 1] = spacing * np.sin(np.arange(3) * angle)
    shape_deform[3, 2] = spacing

    shape_rest = shape_deform.copy()
    shape_rest[3] = [0, 0, 0.022755286528750494]

    np.set_printoptions(suppress=True)
    print('shape diff', shape_deform - shape_rest)

    F = comptute_deformation_gradient(shape_rest, shape_deform)
    print('F', F)

    print('shape_rest_shape_mat')
    print(make_shape_matrix(shape_rest))
    make_shape_matrix(shape_rest[:, [1, 0, 2]]) # check det neg
    print('vtx_norm_rest')
    print(compute_vtx_norm(make_shape_matrix(shape_rest)))
    print('vtx_norm_deform')
    print(compute_vtx_norm(make_shape_matrix(shape_deform)))

    print('cauchy', compute_cauchy_stress(F))

if __name__ == '__main__':
    main()
