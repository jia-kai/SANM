/**
 * \file libsanm/pade.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/pade.h"
#include "libsanm/unary_polynomial.h"

#include <cmath>

using namespace sanm;

PadeApproximation::PadeApproximation(std::span<const TensorND> xs,
                                     bool anm_cond, bool sanity_check)
        : m_sanity_check{sanity_check}, m_xs{xs} {
    sanm_assert(xs.size() >= 3 && xs[0].shape().rank == 1);
    std::unique_ptr<fp_t[]> a_storage{new fp_t[xs.size() * xs.size()]};
    if (xs[0].shape(0) < xs.size() * 2 || xs.size() <= 4) {
        return;
    }
    SANM_SCOPED_PROFILER("pade_build");
    auto a = [p = a_storage.get(), n = xs.size()](int i, int j) -> fp_t& {
        return p[i * n + j];
    };
    int n = xs.size() - 1;

    TensorArray xs_orth(xs.size());
    constexpr fp_t eps = std::numeric_limits<fp_t>::epsilon();

    for (int i = 1; i <= n; ++i) {
        TensorND& uii = xs_orth[i];
        uii = xs[i];

        for (int j = 1; j < i; ++j) {
            a(i, j) = xs[i].flat_dot(xs_orth[j]);
            if (anm_cond && j == 1) {
                // seems to improbve numerical stability
                sanm_assert(fabs(a(i, j)) < 1e-4);
                a(i, j) = 0;
            } else {
                uii.accum_mul(xs_orth[j], -a(i, j));
            }
        }
        fp_t aii = uii.norm_l2();
        if (aii == 0) {
            // exact zero norm. give up
            m_d.clear();
            return;
        }
        a(i, i) = aii;
        uii /= std::max(aii, eps);
        if (aii < eps) {
            uii /= uii.norm_l2();
        }
    }

    if (sanity_check) {
        for (int i = 1; i <= n; ++i) {
            TensorND sum;
            sanm_assert(std::fabs(xs_orth[i].norm_l2() - 1) < 1e-4);
            for (int j = 1; j <= i; ++j) {
                sum.accum_mul(xs_orth[j], a(i, j));
            }
            sum.assert_allclose("pade orth check", xs[i]);
        }
    }

    auto solve_d = [&a, chk = m_sanity_check](std::vector<fp_t>& d, int n) {
        d.resize(n);
        d[0] = 1;
        for (int i = 1; i < n; ++i) {
            fp_t s = 0;
            for (int j = 0; j < i; ++j) {
                s += a(n - j, n - i) * d[j];
            }
            fp_t y = a(n - i, n - i);
            d[i] = -s * y / (y * y + 1e-20);
        }

        if (chk) {
            for (int i = 1; i < n; ++i) {
                fp_t s = 0;
                for (int j = 0; j <= n - i; ++j) {
                    s += a(n - j, i) * d[j];
                }
                sanm_assert(std::fabs(s) < 1e-5, "%g", s);
            }
        }
    };

    solve_d(m_d, n);
    solve_d(m_d_lo, n - 1);

    m_t_nume_coeffs.resize(n, 0);
    for (int i = 0; i < n; ++i) {
        fp_t ti = xs[i].ptr()[xs[i].shape(0) - 1];
        if (!i) {
            m_t0 = ti;
        } else {
            for (int j = 0; j < n - i; ++j) {
                m_t_nume_coeffs[i + j] += m_d[j] * ti;
            }
        }
    }
}

bool PadeApproximation::estimate_valid_range(fp_t start, fp_t eps, fp_t limit) {
    sanm_assert(start > 0 && eps > 0);
    if (m_d.empty()) {
        // rejected in ctor
        return false;
    }
    SANM_SCOPED_PROFILER("pade_est");
    auto roots = unary_polynomial::roots(m_d, true);
    if (!roots.valid()) {
        return false;
    }
    fp_t pole = 0;
    for (auto i : roots.val()) {
        sanm_assert(i.imag() == 0);
        if (i.real() > 0) {
            if (pole == 0 || i.real() < pole) {
                pole = i.real();
            }
        }
    }
    if (pole == 0) {
        pole = start * 4;  // no pole; some arbitrary large value
    }
    if (pole <= start) {
        return false;
    }

    auto check = [this, eps2 = eps * eps,
                  n = static_cast<int>(m_xs.size() - 2)](fp_t a) {
        fp_t denom_n = unary_polynomial::eval(m_d, a),
             denom_lo = unary_polynomial::eval(m_d_lo, a);
        TensorND pn = eval_nume(a, m_d.data(), n),
                 pn_lo = eval_nume(a, m_d_lo.data(), n - 1);
        pn_lo *= denom_n / denom_lo;
        pn_lo -= pn;
        return pn_lo.squared_norm_l2() <= pn.squared_norm_l2() * eps2;
    };
    fp_t left = start * 1.001, right = start + (pole - start) * 0.99;
    if (!check(left)) {
        return false;
    }
    if (limit && right > limit) {
        right = limit;
    }
    if (right > start * 2) {
        // in case pole is too large and no progress is made within iter limit
        if (check(start * 2)) {
            left = start * 2;
        } else {
            right = start * 2;
        }
    }
    int iter = 0;
    while (iter < 8 && right - left > 1e-3) {
        fp_t mid = (left + right) / 2;
        if (check(mid)) {
            left = mid;
        } else {
            right = mid;
        }
        ++iter;
    }
    m_t_max_a = left;
    m_t_max = eval_t(left);

    return true;
}

fp_t PadeApproximation::eval_t(fp_t a) const {
    return unary_polynomial::eval(m_t_nume_coeffs, a) /
                   unary_polynomial::eval(m_d, a) +
           m_t0;
}

TensorND PadeApproximation::eval_nume(fp_t a, const fp_t* d, int n) const {
    TensorND sum = m_xs[0].fill_with(0);
    for (int i = n; i >= 1; --i) {
        sum *= a;
        fp_t scale = unary_polynomial::eval({d, d + n - i + 1}, a);
        sum.accum_mul(m_xs[i], scale);
    }
    return sum;
}

fp_t PadeApproximation::solve_a(fp_t t) const {
    sanm_assert(t >= m_t0 && t <= m_t_max);
    if (t == m_t_max) {
        return m_t_max_a;
    }
    std::vector<fp_t> c = m_t_nume_coeffs;
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] -= (t - m_t0) * m_d[i];
    }
    return unary_polynomial::solve_eqn(c, 0, m_t_max_a, 0);
}

std::pair<TensorND, fp_t> PadeApproximation::eval(fp_t a) const {
    auto y = eval_xt(a);
    TensorND xv = y.take_sub({y.shape(0) - 1});
    fp_t t = y.ptr()[y.shape(0) - 1];
    if (m_sanity_check) {
        fp_t t1 = eval_t(a);
        sanm_assert(std::fabs(t - t1) < 1e-5, "%g vs %g", t, t1);
    }
    return {std::move(xv), t};
}

TensorND PadeApproximation::eval_xt(fp_t a) const {
    TensorND ret = eval_nume(a);
    ret *= a / unary_polynomial::eval(m_d, a);
    ret += m_xs[0];
    return ret;
}
