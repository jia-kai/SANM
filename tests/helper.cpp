/**
 * \file tests/helper.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "tests/helper.h"
#include <random>

using namespace sanm;
using namespace test;

TensorND TensorRNG::operator()(const TensorShape& shape, double low,
                               double high) {
    TensorND ret{shape};
    auto ptr = ret.woptr();
    std::uniform_real_distribution<fp_t> dist{low, high};
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i) {
        ptr[i] = dist(m_rng);
    }
    return ret;
}

void test::require_tensor_eq(const TensorND& a, const TensorND& b,
                             Maybe<std::pair<fp_t, fp_t>> eps_margin,
                             fp_t diff_eps) {
    REQUIRE(a.shape() == b.shape());
    size_t tot = a.shape().total_nr_elems();
    fp_t diff = 0;
    auto p0 = a.ptr(), p1 = b.ptr();
    if (eps_margin.valid()) {
        fp_t eps, margin;
        std::tie(eps, margin) = eps_margin.val();
        for (size_t i = 0; i < tot; ++i) {
            INFO("idx=" << i << " on shape " << a.shape());
            REQUIRE(p0[i] == Approx(p1[i]).margin(margin).epsilon(eps));
            diff += p0[i] - p1[i];
        }
    } else {
        for (size_t i = 0; i < tot; ++i) {
            INFO("idx=" << i << " on shape " << a.shape());
            REQUIRE(p0[i] == Approx(p1[i]));
            diff += p0[i] - p1[i];
        }
    }
    if (diff_eps) {
        REQUIRE(std::fabs(diff / tot) < diff_eps);
    }
}
