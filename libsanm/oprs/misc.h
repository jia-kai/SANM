/**
 * \file libsanm/oprs/misc.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/symbolic.h"

namespace sanm {
namespace symbolic {

class PlaceholderOprMeta final : public OperatorMeta {
    PlaceholderOprMeta() = default;

public:
    const char* name() const override { return "placeholder"; }

    size_t nr_input(OperatorNode*) const override { return 0; }

    size_t nr_output(OperatorNode*) const override { return 1; }

    void on_opr_del(OperatorNode*) const noexcept override {}

    void infer_shape(OperatorNode*, ExecutionContext&) const override;

    void eval_bias(OperatorNode* opr, ExecutionContext& ctx) const override;

    void accum_inp_grad(OperatorNode* opr,
                        ExecutionContext& ctx) const override;

    void compute_order_bias(OperatorNode* opr,
                            ExecutionContext& ctx) const override;

    void compute_coeff(OperatorNode* opr, ExecutionContext& ctx) const override;

    static const PlaceholderOprMeta* instance();
};

class ConstantOprMeta final : public OperatorMeta {
    struct Param {
        TensorND val;
    };
    ConstantOprMeta() = default;

    Param* param(OperatorNode* opr) const {
        sanm_assert(opr->meta() == this);
        return static_cast<Param*>(opr->storage());
    }

public:
    const char* name() const override { return "constant"; }

    size_t nr_input(OperatorNode*) const override { return 0; }

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

    static const ConstantOprMeta* instance();

    static VarNode* make(ComputingGraph& cg, TensorND val);
};

class SliceOprMeta final : public OperatorMeta {
    struct Param {
        int axis, stride;
        Maybe<int> begin, end;
    };
    SliceOprMeta() = default;

    Param* param(OperatorNode* opr) const {
        sanm_assert(opr->meta() == this);
        return static_cast<Param*>(opr->storage());
    }

    //! get the absolute [begin, end) interval (by converting special values)
    std::pair<int, int> abs_interval(OperatorNode* opr, int size) const;

    void compute(OperatorNode* opr, TensorND& dst, const TensorND& src) const;

public:
    const char* name() const override { return "slice"; }

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

    static const SliceOprMeta* instance();

    /*!
     * \brief Taking a subtensor [begin, end) along the given axis with given
     *      stride.
     *
     * Both \p begin and \p end can be negative, which means counting backward
     * from the end.
     *
     * Note: this opr is added to test the Rosenbrock function. Currently we
     * only support axis=1 and stride=1 on batched inputs.
     */
    static VarNode* make(VarNode* x, int axis, Maybe<int> begin, Maybe<int> end,
                         int stride = 1);
};

class ConcatOprMeta final : public OperatorMeta {
    struct Param {
        int nr_input, axis;
    };
    ConcatOprMeta() = default;

    Param* param(OperatorNode* opr) const {
        sanm_assert(opr->meta() == this);
        return static_cast<Param*>(opr->storage());
    }

    void compute(OperatorNode* opr, ExecutionContext& ctx, bool in_coeff) const;

public:
    const char* name() const override { return "concat"; }

    size_t nr_input(OperatorNode* opr) const override {
        return param(opr)->nr_input;
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

    static const ConcatOprMeta* instance();

    /*!
     * \brief Concatenating a few tensors along the given axis.
     *
     * Note: this opr is added to test the Rosenbrock function. Currently we
     * only support axis=1.
     */
    static VarNode* make(std::span<VarNode*> inputs, int axis);
};

}  // namespace symbolic
}  // namespace sanm
