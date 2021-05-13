/**
 * \file libsanm/oprs/reduce.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/symbolic.h"

namespace sanm {
namespace symbolic {
enum class ReduceMode { SUM };

class ReduceOprMeta final : public OperatorMeta {
    struct Param {
        ReduceMode mode;
        int axis;
        bool keepdim;
    };
    ReduceOprMeta() = default;

    static Param* param(OperatorNode* opr) {
        return static_cast<Param*>(opr->storage());
    }

public:
    const char* name() const override { return "reduce"; }

    size_t nr_input(OperatorNode*) const override { return 1; }

    size_t nr_output(OperatorNode*) const override { return 1; }

    void on_opr_del(OperatorNode* opr) const noexcept override {
        delete param(opr);
    }

    void infer_shape(OperatorNode* opr, ExecutionContext& ctx) const override;

    void eval_bias(OperatorNode* opr, ExecutionContext& ctx) const override;

    void accum_inp_grad(OperatorNode* opr,
                        ExecutionContext& ctx) const override;

    void compute_order_bias(OperatorNode* opr,
                            ExecutionContext& ctx) const override;

    void compute_coeff(OperatorNode* opr, ExecutionContext& ctx) const override;

    static const ReduceOprMeta* instance();

    static VarNode* make(VarNode* input, ReduceMode mode, int axis,
                         bool keepdim);
};

}  // namespace symbolic
}  // namespace sanm
