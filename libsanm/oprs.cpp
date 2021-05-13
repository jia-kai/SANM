/**
 * \file libsanm/oprs.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/oprs.h"
#include "libsanm/oprs/analytic_unary.h"
#include "libsanm/oprs/elem_arith.h"
#include "libsanm/oprs/linalg.h"
#include "libsanm/oprs/misc.h"
#include "libsanm/oprs/reduce.h"

using namespace sanm;
using namespace symbolic;

SymbolVar SymbolVar::operator+(SymbolVar rhs) const {
    return linear_combine({{fp_t(1), m_var}, {fp_t(1), rhs.m_var}});
}

SymbolVar SymbolVar::operator-(SymbolVar rhs) const {
    return linear_combine({{fp_t(1), m_var}, {fp_t(-1), rhs.m_var}});
}

SymbolVar SymbolVar::operator*(SymbolVar rhs) const {
    return MultiplyOprMeta::make(m_var, rhs.m_var);
}

SymbolVar SymbolVar::pow(fp_t exp) const {
    if (exp == 1.) {
        return *this;
    }
    return AnalyticUnaryOprMeta::make(node(),
                                      UnaryAnalyticTrait::make_pow(exp));
}

SymbolVar SymbolVar::batched_det() const {
    return BatchDeterminantOprMeta::make(m_var);
}

SymbolVar SymbolVar::reduce_sum(int axis, bool keepdim) const {
    return ReduceOprMeta::make(m_var, ReduceMode::SUM, axis, keepdim);
}

SymbolVar SymbolVar::batched_transpose() const {
    return BatchMatTransposeOprMeta::make(m_var);
}

SymbolVar SymbolVar::batched_matmul(SymbolVar rhs) const {
    return BatchMatMulOprMeta::make(m_var, rhs.node());
}

SymbolVar SymbolVar::batched_mul_eye(size_t dim) const {
    return BatchMulEyeOprMeta::make(m_var, dim);
}

SymbolVar SymbolVar::log() const {
    return AnalyticUnaryOprMeta::make(node(), UnaryAnalyticTrait::make_log());
}

std::array<SymbolVar, 3> SymbolVar::batched_svd_w(bool require_rotation) const {
    auto opr = BatchSVDWOprMeta::make(node(), require_rotation);
    return {opr->output(0), opr->output(1), opr->output(2)};
}

SymbolVar SymbolVar::slice(int axis, Maybe<int> begin, Maybe<int> end,
                           int stride) {
    return SliceOprMeta::make(node(), axis, begin, end, stride);
}

SymbolVar symbolic::linear_combine(
        const std::vector<std::pair<fp_t, SymbolVar>>& vars, fp_t bias) {
    VarNodeArray varptr(vars.size());
    std::vector<fp_t> coeffs(vars.size());
    for (size_t i = 0; i < vars.size(); ++i) {
        coeffs[i] = vars[i].first;
        varptr[i] = vars[i].second.node();
    }
    return LinearCombinationOprMeta::make(std::move(coeffs), std::move(varptr),
                                          bias);
}

SymbolVar symbolic::placeholder(ComputingGraph& cg) {
    return cg.insert_opr(PlaceholderOprMeta::instance(), nullptr, {})
            ->output(0);
}

SymbolVar symbolic::batched_mat_inv_mul(SymbolVar x, SymbolVar a,
                                        bool is_left) {
    return BatchMatInvMulOprMeta::make(x.node(), a.node(), is_left);
}

SymbolVar symbolic::constant(ComputingGraph& cg, TensorND val) {
    return ConstantOprMeta::make(cg, std::move(val));
}

SymbolVar symbolic::concat(std::span<const SymbolVar> xs, int axis) {
    VarNodeArray vx(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        vx[i] = xs[i].node();
    }
    return ConcatOprMeta::make(vx, axis);
}
