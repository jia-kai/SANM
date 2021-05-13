/**
 * \file fea/typedefs.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/typedefs.h"

#include <Eigen/Core>

namespace fea {
using sanm::fp_t;
using sanm::leastsize_t;
using sanm::operator""_fp;
using Vec3 = Eigen::Matrix<fp_t, 3, 1>;
using Mat3 = Eigen::Matrix<fp_t, 3, 3>;
}  // namespace fea
