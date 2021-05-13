/**
 * \file libsanm/unary_polynomial.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/unary_polynomial.h"
#include "libsanm/strio.h"
#include "libsanm/utils.h"

#include <brent.hpp>

#include <cfloat>
#include <cmath>
#include <type_traits>

using namespace sanm;
using namespace unary_polynomial;

namespace {

template <bool neg = false>
class BrentPolyWrapper final : public brent::func_base {
    coeff_t m_coeff;
    fp_t m_bias;

public:
    BrentPolyWrapper(coeff_t coeff, fp_t bias) : m_coeff{coeff}, m_bias{bias} {}

    double operator()(double x) override {
        fp_t y = eval(m_coeff, x) + m_bias;
        if (neg) {
            return -y;
        }
        return y;
    }
};

template <bool is_min>
std::pair<fp_t, fp_t> global_opt(coeff_t f, fp_t xmin, fp_t xmax, fp_t eps) {
    sanm_assert(!f.empty() && xmin < xmax);
    if (f.size() == 1) {
        return {xmin, f[0]};
    }
    auto xinit = ((f[1] < 0) ^ is_min) ? xmin : xmax;
    if (f.size() == 2) {
        return {xinit, eval(f, xinit)};
    }
    fp_t m = 0, xabs = std::max(std::fabs(xmin), std::fabs(xmax));
    for (size_t i = f.size() - 1; i >= 2; --i) {
        m = m * xabs + i * (i - 1) * std::fabs(f[i]);
    }
    double xs = xmin - 1;
    BrentPolyWrapper<is_min ? false : true> wrapper{f, 0};
    double y = brent::glomin(xmin, xmax, xinit, m, eps, eps, wrapper, xs);
    if (xs < xmin || xs > xmax) {
        // this brent library sometimes fails
        xs = xinit;
        y = wrapper(xs);
    }
    if (fp_t y0 = wrapper(xinit); y0 < y) {
        xs = xinit;
        y = y0;
    }
    if (!is_min) {
        y = -y;
    }
    return {xs, y};
}
}  // anonymous namespace

fp_t unary_polynomial::eval(coeff_t f, fp_t x) {
    fp_t ret = 0;
    for (int i = f.size() - 1; i >= 0; --i) {
        ret = ret * x + f[i];
    }
    return ret;
}

fp_t unary_polynomial::solve_quad(fp_t a, fp_t b, fp_t c) {
    sanm_assert(a > 0, "bad a: %g", a);
    fp_t delta = b * b - 4 * a * c;
    if (delta < 0) {
        return -b / (a * 2);
    }
    return (std::sqrt(delta) - b) / (a * 2);
}

fp_t unary_polynomial::solve_eqn(coeff_t f, fp_t xmin, fp_t xmax, fp_t b,
                                 fp_t eps) {
    sanm_assert(!f.empty() && xmin < xmax);
    BrentPolyWrapper wrapper{f, -b};
    auto f0 = wrapper(xmin), f1 = wrapper(xmax);
    sanm_assert(f0 * f1 <= 0, "no zero point: f0=%g f1=%g", f0, f1);
    return brent::zero(xmin, xmax, eps, wrapper);
}

fp_t unary_polynomial::stable_x_range(int order) {
    if constexpr (std::is_same_v<fp_t, double>) {
        // about 15.9 decimal digits in double precision
        return std::pow(1e15, 1.0 / static_cast<double>(order));
    }
    static_assert(std::is_same_v<fp_t, double>);
}

std::pair<fp_t, fp_t> unary_polynomial::minimize(coeff_t f, fp_t xmin,
                                                 fp_t xmax, fp_t eps) {
    return global_opt<true>(f, xmin, xmax, eps);
}

std::pair<fp_t, fp_t> unary_polynomial::maximize(coeff_t f, fp_t xmin,
                                                 fp_t xmax, fp_t eps) {
    return global_opt<false>(f, xmin, xmax, eps);
}

TensorND unary_polynomial::eval_tensor(std::span<const TensorND> f, fp_t x) {
    TensorND ret;
    for (int i = f.size() - 1; i >= 0; --i) {
        if (ret.empty()) {
            ret = f[i];
        } else {
            ret *= x;
            ret += f[i];
        }
    }
    return ret;
}

/*******************************************************************************
 * FindPolynomialRoots
 *
 * The Bairstow and Newton correction formulae are used for a simultaneous
 * linear and quadratic iterated synthetic division.  The coefficients of
 * a polynomial of degree n are given as a[i] (i=0,i,..., n) where a[0] is
 * the constant term.  The coefficients are scaled by dividing them by
 * their geometric mean.  The Bairstow or Newton iteration method will
 * nearly always converge to the number of figures carried, fig, either to
 * root values or to their reciprocals.  If the simultaneous Newton and
 * Bairstow iteration fails to converge on root values or their
 * reciprocals in maxiter iterations, the convergence requirement will be
 * successively reduced by one decimal figure.  This program anticipates
 * and protects against loss of significance in the quadratic synthetic
 * division.  (Refer to "On Programming the Numerical Solution of
 * Polynomial Equations," by K. W. Ellenberger, Commun. ACM 3 (Dec. 1960),
 * 644-647.)  The real and imaginary part of each root is stated as u[i]
 * and v[i], respectively.
 *
 * ACM algorithm #30 - Numerical Solution of the Polynomial Equation
 * K. W. Ellenberger
 * Missle Division, North American Aviation, Downey, California
 * Converted to C, modified, optimized, and structured by
 * Ken Turkowski
 * CADLINC, Inc., Palo Alto, California
 *******************************************************************************/
Maybe<std::vector<std::complex<fp_t>>> unary_polynomial::roots(coeff_t a,
                                                               bool only_real,
                                                               int max_iter,
                                                               fp_t tol) {
    sanm_assert(a.size() >= 2);
    int n = a.size() - 1;
    int i;
    int j;
    std::unique_ptr<fp_t[]> workspace{new fp_t[(n + 3) * 5]};
    fp_t *h = workspace.get(), *b = h + n + 3, *c = b + n + 3, *d = c + n + 3,
         *e = d + n + 3;
    // [-2 : n]
    fp_t K, ps, qs, pt, qt, s, rev, r = 0;
    int t;
    fp_t p = 0, q = 0, qq;

    // Zero elements with negative indices
    b[2 + -1] = b[2 + -2] = c[2 + -1] = c[2 + -2] = d[2 + -1] = d[2 + -2] =
            e[2 + -1] = e[2 + -2] = h[2 + -1] = h[2 + -2] = 0.0;

    // Copy polynomial coefficients to working storage
    {
        int m = 0;
        for (j = n; j >= 0; j--)
            h[2 + j] = a[m++];  // Note reversal of coefficients
    }

    t = 1;
    K = 1 / tol;  // Relative accuracy

    std::vector<std::complex<fp_t>> roots;
    for (; h[2 + n] == 0.0; n--) {  // Look for zero high-order coeff.
        sanm_assert(n > -2);
    }

INIT:
    if (n == 0)
        return roots;

    ps = qs = pt = qt = s = 0.0;
    rev = 1.0;

    if (n == 1) {
        r = -h[2 + 1] / h[2 + 0];
        goto LINEAR;
    }

    for (j = n; j >= 0; j--)  // Find geometric mean of coeff's
        if (h[2 + j] != 0.0)
            s += log(fabs(h[2 + j]));
    s = exp(s / (n + 1));

    for (j = n; j >= 0; j--)  // Normalize coeff's by mean
        h[2 + j] /= s;

    if (fabs(h[2 + 1] / h[2 + 0]) < fabs(h[2 + n - 1] / h[2 + n])) {
    REVERSE:
        t = -t;
        for (j = (n - 1) / 2; j >= 0; j--) {
            s = h[2 + j];
            h[2 + j] = h[2 + n - j];
            h[2 + n - j] = s;
        }
    }
    if (qs != 0.0) {
        p = ps;
        q = qs;
    } else {
        if (h[2 + n - 2] == 0.0) {
            q = 1.0;
            p = -2.0;
        } else {
            q = h[2 + n] / h[2 + n - 2];
            p = (h[2 + n - 1] - q * h[2 + n - 3]) / h[2 + n - 2];
        }
        if (n == 2)
            goto QADRTIC;
        r = 0.0;
    }
ITERATE:
    for (i = max_iter; i > 0; i--) {
        for (j = 0; j <= n; j++) {  // BAIRSTOW
            b[2 + j] = h[2 + j] - p * b[2 + j - 1] - q * b[2 + j - 2];
            c[2 + j] = b[2 + j] - p * c[2 + j - 1] - q * c[2 + j - 2];
        }
        if ((h[2 + n - 1] != 0.0) && (b[2 + n - 1] != 0.0)) {
            if (fabs(h[2 + n - 1] / b[2 + n - 1]) >= K) {
                b[2 + n] = h[2 + n] - q * b[2 + n - 2];
            }
            if (b[2 + n] == 0.0)
                goto QADRTIC;
            if (K < fabs(h[2 + n] / b[2 + n]))
                goto QADRTIC;
        }

        for (j = 0; j <= n; j++) {  // NEWTON
            d[2 + j] =
                    h[2 + j] + r * d[2 + j - 1];  // Calculate polynomial at r
            e[2 + j] =
                    d[2 + j] + r * e[2 + j - 1];  // Calculate derivative at r
        }
        if (d[2 + n] == 0.0)
            goto LINEAR;
        if (K < fabs(h[2 + n] / d[2 + n]))
            goto LINEAR;

        c[2 + n - 1] = -p * c[2 + n - 2] - q * c[2 + n - 3];
        s = c[2 + n - 2] * c[2 + n - 2] - c[2 + n - 1] * c[2 + n - 3];
        if (s == 0.0) {
            p -= 2.0;
            q *= (q + 1.0);
        } else {
            p += (b[2 + n - 1] * c[2 + n - 2] - b[2 + n] * c[2 + n - 3]) / s;
            q += (-b[2 + n - 1] * c[2 + n - 1] + b[2 + n] * c[2 + n - 2]) / s;
        }
        if (e[2 + n - 1] == 0.0)
            r -= 1.0;  // Minimum step
        else
            r -= d[2 + n] / e[2 + n - 1];  // Newton's iteration
    }
    ps = pt;
    qs = qt;
    pt = p;
    qt = q;
    if (rev < 0.0)
        K /= 10.0;
    if (K < 1e-8) {
        return None;
    }
    rev = -rev;
    goto REVERSE;

LINEAR:
    if (t < 0)
        r = 1.0 / r;
    n--;
    roots.emplace_back(r, 0._fp);

    for (j = n; j >= 0; j--) {  // Polynomial deflation by lin-nomial
        if ((d[2 + j] != 0.0) && (fabs(h[2 + j] / d[2 + j]) < K))
            h[2 + j] = d[2 + j];
        else
            h[2 + j] = 0.0;
    }

    if (n == 0)
        return roots;
    goto ITERATE;

QADRTIC:
    if (t < 0) {
        p /= q;
        q = 1.0 / q;
    }
    n -= 2;

    if (0.0 < (q - (p * p / 4.0))) {  // Two complex roots
        s = sqrt(q - (p * p / 4.0));
        if (!only_real) {
            roots.emplace_back(-p / 2._fp, s);
            roots.emplace_back(-p / 2._fp, -s);
        }
    } else {  // Two real roots
        s = sqrt(((p * p / 4.0)) - q);
        if (p < 0.0) {
            qq = -p / 2.0 + s;
        } else {
            qq = -p / 2.0 - s;
        }
        roots.emplace_back(qq, 0._fp);
        roots.emplace_back(q / qq, 0._fp);
    }

    for (j = n; j >= 0; j--) {  // Polynomial deflation by quadratic
        if ((b[2 + j] != 0.0) && (fabs(h[2 + j] / b[2 + j]) < K))
            h[2 + j] = b[2 + j];
        else
            h[2 + j] = 0.0;
    }
    goto INIT;
}
