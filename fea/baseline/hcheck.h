#pragma once

#include "material.h"

namespace materials {
void check_hessian(const char* name, const Material<3, double>& material);
}  // namespace materials
