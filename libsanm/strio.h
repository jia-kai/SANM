/**
 * \file libsanm/strio.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// string IO helpers
#pragma once

#include <sstream>
#include <vector>

namespace sanm {
template <typename T>
std::string str(const T& val) {
    std::ostringstream ostr;
    ostr << val;
    return ostr.str();
}

template <typename T>
std::string str(const std::vector<T>& val) {
    std::ostringstream ostr;
    ostr << "{";
    for (size_t i = 0; i < val.size(); ++i) {
        if (i) {
            ostr << ",";
        }
        ostr << val[i];
    }
    ostr << "}";
    return ostr.str();
}
}  // namespace sanm
