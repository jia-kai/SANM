/**
 * \file tests/pade.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// test functions related to Pade approximation
#include "libsanm/pade.h"
#include "libsanm/unary_polynomial.h"
#include "tests/helper.h"

#include <cmath>

using namespace sanm;
using namespace test;

TEST_CASE("Pade.UnaryPolynomialRoots") {
    constexpr size_t N = 10;
    TensorRNG rng;
    TensorND cf0 = rng({N - 2}), coeffs{{N}};
    cf0.rwptr()[N - 3] = 2.3;
    auto cptr = coeffs.rwptr();
    for (size_t i = 0; i < N; ++i) {
        // conv with (x - 3) * (x + 4)
        fp_t rhs[] = {-12, 1, 1}, s = 0;
        for (size_t j = 0; j < 3; ++j) {
            if (j <= i && i - j < N - 2) {
                s += rhs[j] * cf0.ptr()[i - j];
            }
        }
        cptr[i] = s;
    }

    auto roots = unary_polynomial::roots({cptr, cptr + N}, false).val();
    std::vector<fp_t> r_roots;
    REQUIRE(roots.size() == N - 1);
    printf("roots:");
    for (size_t i = 0; i < N - 1; ++i) {
        std::complex<fp_t> ri = roots[i], s{0, 0};
        if (ri.imag() == 0) {
            r_roots.emplace_back(ri.real());
        }
        printf(" %.2g+%.2gi", ri.real(), ri.imag());
        for (int j = N - 1; j >= 0; --j) {
            s = s * ri + cptr[j];
        }
        REQUIRE(s.real() == Approx(0).margin(2e-4));
        REQUIRE(s.imag() == Approx(0).margin(2e-4));
    }
    printf("  nr_real=%zu\n", r_roots.size());
    REQUIRE(r_roots.size() >= 2);

    auto roots_real = unary_polynomial::roots({cptr, cptr + N}, true).val();
    REQUIRE(roots_real.size() == r_roots.size());
    std::sort(r_roots.begin(), r_roots.end());
    std::sort(roots_real.begin(), roots_real.end(),
              [](std::complex<fp_t> a, std::complex<fp_t> b) {
                  return a.real() < b.real();
              });
    for (size_t i = 0; i < r_roots.size(); ++i) {
        REQUIRE(roots_real[i].real() == Approx(r_roots[i]));
    }
}

TEST_CASE("Pade.Approx") {
    constexpr int SIZE = 500, N = 9;
    constexpr fp_t eps = 1e-5;
    TensorRNG rng;
    TensorArray xs;
    for (int i = 0; i < N; ++i) {
        xs.emplace_back(rng({SIZE}) *= std::pow(0.5, i + 1));
    }
    xs[1].rwptr()[SIZE - 1] = 2.3;

    auto range0 = std::pow(eps * xs[1].norm_l2() / xs[N - 1].norm_l2(),
                           1.0 / (N - 2));
    PadeApproximation pade{xs, false, true};
    REQUIRE(pade.estimate_valid_range(range0 / 10, eps));
    printf("Pade ranges: %g %g\n", range0, pade.get_t_max_a());

    fp_t tmin = xs[0].rwptr()[SIZE - 1], tmax = pade.get_t_max();
    REQUIRE(tmax > tmin);

    for (fp_t div : {8._fp, 3._fp, 1.01_fp}) {
        fp_t a = pade.get_t_max_a() / div;
        auto expect = unary_polynomial::eval_tensor(xs, a);
        auto get = pade.eval(a);
        require_tensor_eq(expect.take_sub({SIZE - 1}), get.first,
                          std::make_pair(1e-4_fp, 1e-4_fp));
        REQUIRE(expect.ptr()[SIZE - 1] == Approx(get.second));
    }

    for (fp_t frac : {1e-3_fp, 0.27_fp, 0.96_fp}) {
        fp_t t = tmin * (1 - frac) + tmax * frac, a = pade.solve_a(t);
        auto expect = unary_polynomial::eval_tensor(xs, a);
        auto get = pade.eval(a);
        require_tensor_eq(expect.take_sub({SIZE - 1}), get.first,
                          std::make_pair(1e-4_fp, 1e-4_fp));
        REQUIRE(expect.ptr()[SIZE - 1] == Approx(get.second));
        REQUIRE(get.second == Approx(t));
    }
}
