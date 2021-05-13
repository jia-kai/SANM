/**
 * \file libsanm/oprs/reduce.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/oprs/reduce.h"

using namespace sanm;
using namespace symbolic;

void ReduceOprMeta::infer_shape(OperatorNode* opr,
                                ExecutionContext& ctx) const {
    auto param = this->param(opr);
    const TensorShape& ishp = ctx.get(opr->input(0)).shape;
    ctx.get(opr->output(0)).shape =
            infer_reduce_shape(ishp, param->axis, param->keepdim);
}

void ReduceOprMeta::eval_bias(OperatorNode* opr, ExecutionContext& ctx) const {
    auto&& ictx = ctx.get(opr->input(0));
    auto&& octx = ctx.get(opr->output(0));
    auto param = this->param(opr);
    octx.coeffs.emplace_back(
            ictx.coeffs[0].reduce_sum(param->axis, param->keepdim));
}

void ReduceOprMeta::accum_inp_grad(OperatorNode* opr,
                                   ExecutionContext& ctx) const {
    auto& ictx = ctx.get(opr->input(0));
    const auto& octx = ctx.get(opr->output(0));
    auto param = this->param(opr);

    size_t bcast_size, before = 1, after = 1;
    if (param->axis == -1) {
        sanm_assert(octx.jacobian.is_batched(),
                    "reduce axis=-1 with batched out grad");
        bcast_size = ictx.shape.total_nr_elems_per_batch();
    } else if (param->axis == -2) {
        sanm_assert(!octx.jacobian.is_batched());
        bcast_size = ictx.shape.total_nr_elems();
    } else {
        for (int i = octx.jacobian.is_batched() ? 1 : 0; i < param->axis; ++i) {
            before *= ictx.shape[i];
        }
        bcast_size = ictx.shape[param->axis];
        for (size_t i = param->axis + 1; i < ictx.shape.rank; ++i) {
            after *= ictx.shape[i];
        }
    }

    const size_t graph_odim = octx.jacobian.out_dim();
    TensorND gy = octx.jacobian.as_full();
    TensorShape gx_shape;
    if (octx.jacobian.is_batched()) {
        sanm_assert(param->axis != 0);
        const size_t batch = octx.jacobian.batch();
        gy.reshape_inplace({batch * graph_odim * before, 1, after});
        gx_shape = {batch, graph_odim, before * bcast_size * after};
    } else {
        gy.reshape_inplace({graph_odim * before, 1, after});
        gx_shape = {graph_odim, before * bcast_size * after};
    }
    sanm_assert(param->mode == ReduceMode::SUM);
    TensorND gx;
    gx.as_broadcast(gy, 1, bcast_size).reshape_inplace(gx_shape);
    ictx.jacobian.accum_full(octx.jacobian.is_batched(), gx);
}

void ReduceOprMeta::compute_order_bias(OperatorNode* opr,
                                       ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    auto param = this->param(opr);
    sanm_assert(param->mode == ReduceMode::SUM);
    octx.cur_order_bias.as_reduce_sum(ictx.cur_order_bias, param->axis,
                                      param->keepdim);
}

void ReduceOprMeta::compute_coeff(OperatorNode* opr,
                                  ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    auto param = this->param(opr);
    sanm_assert(param->mode == ReduceMode::SUM);
    octx.coeffs.emplace_back().as_reduce_sum(ictx.coeffs.back(), param->axis,
                                             param->keepdim);
}

const ReduceOprMeta* ReduceOprMeta::instance() {
    static ReduceOprMeta inst;
    return &inst;
}

VarNode* ReduceOprMeta::make(VarNode* input, ReduceMode mode, int axis,
                             bool keepdim) {
    sanm_assert(axis != 0, "can not reduce on batch dim");
    std::unique_ptr<Param> param{new Param{mode, axis, keepdim}};
    auto opr =
            input->owner_graph()->insert_opr(instance(), param.get(), {input});
    param.release();
    return opr->output(0);
}
