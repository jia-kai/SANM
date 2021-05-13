/**
 * \file libsanm/oprs/elem_arith.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// elementary arithmetic in element-wise manner; each input var can either be
// the scalar or have the full shape
#pragma once

#include "libsanm/symbolic.h"

namespace sanm {
namespace symbolic {

class LinearCombinationOprMeta final : public OperatorMeta {
    struct Param {
        std::vector<fp_t> coeffs;
        fp_t bias;
    };

    Param* param(OperatorNode* opr) const {
        sanm_assert(opr->meta() == this);
        return static_cast<Param*>(opr->storage());
    }

    LinearCombinationOprMeta() = default;

    void compute_bias(OperatorNode* opr, ExecutionContext& ctx,
                      bool in_coeff) const;

public:
    const char* name() const override { return "linear_combination"; }

    size_t nr_input(OperatorNode* opr) const override {
        return param(opr)->coeffs.size();
    }

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

    static const LinearCombinationOprMeta* instance();

    static VarNode* make(std::vector<fp_t> coeffs, VarNodeArray inputs,
                         fp_t bias);
};

class MultiplyOprMeta final : public OperatorMeta {
    MultiplyOprMeta() = default;

    struct Userdata final : public VarNodeExeCtx::Userdata {
        TensorND self_bias;
    };

public:
    const char* name() const override { return "multiply"; }

    size_t nr_input(OperatorNode* opr) const override { return 2; }

    size_t nr_output(OperatorNode*) const override { return 1; }

    void on_opr_del(OperatorNode*) const noexcept override {}

    void infer_shape(OperatorNode* opr, ExecutionContext& ctx) const override;

    void eval_bias(OperatorNode* opr, ExecutionContext& ctx) const override;

    void accum_inp_grad(OperatorNode* opr,
                        ExecutionContext& ctx) const override;

    void compute_order_bias(OperatorNode* opr,
                            ExecutionContext& ctx) const override;

    void compute_coeff(OperatorNode* opr, ExecutionContext& ctx) const override;
    static const MultiplyOprMeta* instance();

    static VarNode* make(VarNode* x, VarNode* y);
};

}  // namespace symbolic
}  // namespace sanm
