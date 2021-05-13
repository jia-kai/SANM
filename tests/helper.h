/**
 * \file tests/helper.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/tensor.h"

#include <catch2/catch.hpp>

namespace sanm {
namespace test {

class TensorRNG {
    double m_low = -1, m_high = 1;
    Xorshift128pRng m_rng{92702102};

public:
    TensorRNG() = default;
    TensorRNG(double low, double high) : m_low{low}, m_high{high} {}

    TensorND operator()(const TensorShape& shape) {
        return operator()(shape, m_low, m_high);
    }

    TensorND operator()(const TensorShape& shape, double low, double high);

    Xorshift128pRng& raw_rng() { return m_rng; }
};

/*!
 * \brief asserts that the two tensors have the same shape and are close to each
 *      other
 * \param diff_eps check that abs((a-b).mean()) <= diff_eps if it is not zero
 */
void require_tensor_eq(const TensorND& a, const TensorND& b,
                       Maybe<std::pair<fp_t, fp_t>> eps_margin = None,
                       fp_t diff_eps = 0);

}  // namespace test

static inline std::ostream& operator<<(std::ostream& ostr,
                                       const TensorShape& shape) {
    return ostr << shape.str();
}
}  // namespace sanm
