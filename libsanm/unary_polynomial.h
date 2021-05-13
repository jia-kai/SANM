/**
 * \file libsanm/unary_polynomial.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

//! utilities to work with unary polynomials
#pragma once

#include "libsanm/tensor.h"

#include <array>
#include <complex>
#include <span>

namespace sanm {
//! helpers for dealing with polynomials with only one variable
namespace unary_polynomial {
using coeff_t = std::span<const fp_t>;

//! evalulate f(x)
fp_t eval(coeff_t f, fp_t x);

//! solve the quadratic equation a*x^2 + b*x + c = 0; the minimizer would be
//! returned if there is no solution. Require a > 0
fp_t solve_quad(fp_t a, fp_t b, fp_t c);

//! solve x such that f(x) = b
fp_t solve_eqn(coeff_t f, fp_t xmin, fp_t xmax, fp_t b = 0, fp_t eps = 1e-6);

//! range of x for a polynomial so that the evaluation is numerically stable
fp_t stable_x_range(int order);

/*!
 * \brief compute the global minimum of \p f
 * \return the minimizer x*, and corresponding f(x*)
 */
std::pair<fp_t, fp_t> minimize(coeff_t f, fp_t xmin, fp_t xmax,
                               fp_t eps = 1e-6);

//! see minimize()
std::pair<fp_t, fp_t> maximize(coeff_t f, fp_t xmin, fp_t xmax,
                               fp_t eps = 1e-6);

/*!
 * \brief find all roots of a polynomial
 *
 * \param only_real whether to only return real roots
 * \return the roots, or None if the algorithm does not converge
 */
Maybe<std::vector<std::complex<fp_t>>> roots(coeff_t f, bool only_real,
                                             int max_iter = 300,
                                             fp_t tol = 1e-8);

//! compute sum(f[i]*x^i)
TensorND eval_tensor(std::span<const TensorND> f, fp_t x);

}  // namespace unary_polynomial
}  // namespace sanm
