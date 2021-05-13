/**
 * \file libsanm/typedefs.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include <cstdint>

namespace sanm {
//! floating point type
using fp_t = double;

//! A type similar to size_t to be used in storage. We use 32-bit integers
//! because they are big enough while allowing more efficient cache on 64-bit
//! systems
using leastsize_t = uint32_t;

static inline constexpr fp_t operator"" _fp(long double v) {
    return v;
}
}  // namespace sanm
