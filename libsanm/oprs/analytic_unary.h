/**
 * \file libsanm/oprs/analytic_unary.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// unary analytic functions
#pragma once
#include "libsanm/analytic_unary.h"
#include "libsanm/symbolic.h"

namespace sanm {
namespace symbolic {

class AnalyticUnaryOprMeta final : public OperatorMeta {
    struct StateCache;
    struct Param {
        UnaryAnalyticTraitPtr trait;
    };

    Param* param(OperatorNode* opr) const {
        sanm_assert(opr->meta() == this);
        return static_cast<Param*>(opr->storage());
    }

    AnalyticUnaryOprMeta() = default;

public:
    const char* name() const override { return "unary_analytic"; }

    size_t nr_input(OperatorNode*) const override { return 1; }

    size_t nr_output(OperatorNode*) const override { return 1; }

    void on_opr_del(OperatorNode* opr) const noexcept override {
        delete param(opr);
    }

    void infer_shape(OperatorNode* opr, ExecutionContext& ctx) const override {
        ctx.get(opr->output(0)).shape = ctx.get(opr->input(0)).shape;
    }

    void eval_bias(OperatorNode* opr, ExecutionContext& ctx) const override;

    void accum_inp_grad(OperatorNode* opr,
                        ExecutionContext& ctx) const override;

    void compute_order_bias(OperatorNode* opr,
                            ExecutionContext& ctx) const override;

    void compute_coeff(OperatorNode* opr, ExecutionContext& ctx) const override;

    static const AnalyticUnaryOprMeta* instance();
    static VarNode* make(VarNode* x, UnaryAnalyticTraitPtr trait);
};

}  // namespace symbolic
}  // namespace sanm
