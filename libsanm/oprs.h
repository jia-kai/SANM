/**
 * \file libsanm/oprs.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/symbolic.h"

namespace sanm {
namespace symbolic {

//! wrapper of VarNode* to overload some operators and add arithmetic functions
class SymbolVar {
    VarNode* m_var = nullptr;

public:
    SymbolVar() = default;
    SymbolVar(VarNode* var) : m_var{var} {}

    VarNode* node() const { return m_var; }

    SymbolVar operator+(SymbolVar rhs) const;
    inline SymbolVar operator+(fp_t rhs) const;

    SymbolVar operator-(SymbolVar rhs) const;
    SymbolVar operator-(fp_t rhs) const { return *this + (-rhs); }

    SymbolVar operator*(SymbolVar rhs) const;
    inline SymbolVar operator*(fp_t rhs) const;

    //! see TensorND::reduce_sum()
    SymbolVar reduce_sum(int axis, bool keepdim = true) const;

    //! batched matrix transpose
    SymbolVar batched_transpose() const;

    //! batched matrix inverse
    inline SymbolVar batched_matinv() const;

    //! batched matrix multiplication with another variable
    SymbolVar batched_matmul(SymbolVar rhs) const;

    //! compute the batched determinants of matrices in the input tensor
    SymbolVar batched_det() const;

    //! multiply this scalar with an identity matrix of given size
    SymbolVar batched_mul_eye(size_t dim) const;

    //! elementwise power with constant exponent
    SymbolVar pow(fp_t exp) const;

    //! elementwise natural logarithm
    SymbolVar log() const;

    //! see TensorND::batched_svd_w
    std::array<SymbolVar, 3> batched_svd_w(bool require_rotation = false) const;

    //! see SliceOprMeta::make
    SymbolVar slice(int axis, Maybe<int> begin, Maybe<int> end, int stride = 1);
};

/*!
 * \brief compute y such that y @ x = a
 * \param a if it is empty, the identity matrix would be used
 * \param is_left use y @ x if true and x @ y if false
 */
SymbolVar batched_mat_inv_mul(SymbolVar x, SymbolVar a, bool is_left);

//! a linear combination of given vars, which must either be batched scalars or
//! have the same shape as the output
SymbolVar linear_combine(const std::vector<std::pair<fp_t, SymbolVar>>& vars,
                         fp_t bias = 0);

//! a placeholder to represent an input variable
SymbolVar placeholder(ComputingGraph& cg);

/*!
 * \brief a variable with a constant value;
 *
 * The value is assumed to be batched if used in the context of
 * ParallelTaylorCoeffProp.
 */
SymbolVar constant(ComputingGraph& cg, TensorND val);

SymbolVar SymbolVar::batched_matinv() const {
    return batched_mat_inv_mul(*this, {}, true);
}

SymbolVar SymbolVar::operator+(fp_t rhs) const {
    return linear_combine({{1._fp, *this}}, rhs);
}

SymbolVar SymbolVar::operator*(fp_t rhs) const {
    return linear_combine({{rhs, *this}}, 0);
}

static inline SymbolVar operator-(fp_t x, SymbolVar y) {
    return linear_combine({{-1._fp, y}}, x);
}

//! concat along an axis; see also ConcatOprMeta::make
SymbolVar concat(std::span<const SymbolVar> xs, int axis);

}  // namespace symbolic

using SymbolVar = symbolic::SymbolVar;

}  // namespace sanm
