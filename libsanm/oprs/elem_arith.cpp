/**
 * \file libsanm/oprs/elem_arith.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/oprs/elem_arith.h"

using namespace sanm;
using namespace symbolic;

namespace {
//! infer output shape from input shapes; we only allow broadcasting scalars
void infer_shape_elemwise(OperatorNode* opr, ExecutionContext& ctx) {
    TensorShape& oshp = ctx.get(opr->output(0)).shape;
    sanm_assert(oshp.rank == 0);
    for (VarNode* i : opr->inputs()) {
        const TensorShape& ishp = ctx.get(i).shape;
        if (!oshp.rank) {
            oshp = ishp;
        } else if (oshp != ishp) {
            sanm_assert(oshp.is_batched_scalar() || ishp.is_batched_scalar(),
                        "invalid shape in elem arith: %s vs %s",
                        oshp.str().c_str(), ishp.str().c_str());
            if (ishp.is_single_scalar()) {
                // it can be broadcasted
                continue;
            }
            if (oshp.is_single_scalar() ||
                (oshp.is_batched_scalar() && !ishp.is_batched_scalar())) {
                oshp = ishp;
            }
            sanm_assert(oshp[0] == ishp[0],
                        "batch mismatch in elem arith: %s vs %s",
                        oshp.str().c_str(), ishp.str().c_str());
        }
    }
}
}  // anonymous namespace

/* ======================= LinearCombinationOprMeta ======================= */

void LinearCombinationOprMeta::infer_shape(OperatorNode* opr,
                                           ExecutionContext& ctx) const {
    infer_shape_elemwise(opr, ctx);
}

void LinearCombinationOprMeta::eval_bias(OperatorNode* opr,
                                         ExecutionContext& ctx) const {
    auto param = this->param(opr);
    TensorND result{ctx.get(opr->output(0)).shape};
    result.fill_with_inplace(param->bias);
    for (size_t i = 0; i < param->coeffs.size(); ++i) {
        result.accum_mul(ctx.get(opr->input(i)).coeffs[0], param->coeffs[i]);
    }
    ctx.get(opr->output(0)).coeffs.emplace_back(std::move(result));
}

void LinearCombinationOprMeta::accum_inp_grad(OperatorNode* opr,
                                              ExecutionContext& ctx) const {
    const auto& octx = ctx.get(opr->output(0));
    TensorND gout_full;
    auto param = this->param(opr);
    for (size_t i = 0; i < opr->inputs().size(); ++i) {
        auto& ictx = ctx.get(opr->input(i));
        fp_t k = param->coeffs[i];
        if (ictx.shape == octx.shape) {
            ictx.jacobian += octx.jacobian.compose_with_scaling(k);
        } else {
            sanm_assert(ictx.shape.is_batched_scalar());
            sanm_assert(octx.jacobian.is_batched() ==
                                !ictx.shape.is_single_scalar(),
                        "unimplemented case");
            if (gout_full.empty()) {
                gout_full = octx.jacobian.as_full();
                sanm_assert(gout_full.rank() == 2 || gout_full.rank() == 3);
            }
            ictx.jacobian.accum_full(
                    octx.jacobian.is_batched(),
                    gout_full.reduce_sum(gout_full.rank() - 1, true) * k);
        }
    }
}

void LinearCombinationOprMeta::compute_order_bias(OperatorNode* opr,
                                                  ExecutionContext& ctx) const {
    compute_bias(opr, ctx, false);
}

void LinearCombinationOprMeta::compute_bias(OperatorNode* opr,
                                            ExecutionContext& ctx,
                                            bool in_coeff) const {
    auto& octx = ctx.get(opr->output(0));
    auto param = this->param(opr);
    TensorND& dst = octx.get_bias(in_coeff);
    dst.set_shape({octx.shape}).fill_with_inplace(0);
    for (size_t i = 0; i < opr->inputs().size(); ++i) {
        auto& ictx = ctx.get(opr->input(i));
        dst.accum_mul(ictx.get_bias(in_coeff), param->coeffs[i]);
    }
}

void LinearCombinationOprMeta::compute_coeff(OperatorNode* opr,
                                             ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    octx.coeffs.emplace_back();
    compute_bias(opr, ctx, true);
}

const LinearCombinationOprMeta* LinearCombinationOprMeta::instance() {
    static LinearCombinationOprMeta inst;
    return &inst;
}

VarNode* LinearCombinationOprMeta::make(std::vector<fp_t> coeffs,
                                        VarNodeArray inputs, fp_t bias) {
    sanm_assert(!inputs.empty() && inputs.size() == coeffs.size());
    std::unique_ptr<Param> param{new Param{std::move(coeffs), bias}};
    auto ret = inputs[0]
                       ->owner_graph()
                       ->insert_opr(instance(), param.get(), std::move(inputs))
                       ->output(0);
    param.release();
    return ret;
}

/* ======================= MultiplyOprMeta ======================= */

void MultiplyOprMeta::infer_shape(OperatorNode* opr,
                                  ExecutionContext& ctx) const {
    infer_shape_elemwise(opr, ctx);
}

void MultiplyOprMeta::eval_bias(OperatorNode* opr,
                                ExecutionContext& ctx) const {
    ctx.get(opr->output(0))
            .coeffs.emplace_back(ctx.get(opr->input(0)).coeffs[0] *
                                 ctx.get(opr->input(1)).coeffs[0]);
}

void MultiplyOprMeta::accum_inp_grad(OperatorNode* opr,
                                     ExecutionContext& ctx) const {
    const auto& octx = ctx.get(opr->output(0));
    VarNodeExeCtx* ictx[2] = {&ctx.get(opr->input(0)), &ctx.get(opr->input(1))};
    for (int i = 0; i < 2; ++i) {
        if (ictx[i]->shape != octx.shape) {
            sanm_assert(octx.jacobian.is_batched() ==
                                !ictx[i]->shape.is_single_scalar(),
                        "unimplemented case");
        }
    }
    for (int i = 0; i < 2; ++i) {
        TensorND other = ictx[!i]->coeffs[0];
        if (octx.jacobian.is_batched()) {
            other = other.flatten_batched();
        } else {
            other = other.flatten_as_vec();
        }
        if (octx.jacobian.type() == StSparseLinearTrans::FULL) {
            other.reshape_inplace(other.shape().add_axis(other.rank() - 1));
        }
        StSparseLinearTrans gi{octx.jacobian.type(), octx.jacobian.is_batched(),
                               octx.jacobian.coeff() * other};
        if (ictx[i]->shape == octx.shape) {
            ictx[i]->jacobian += gi;
        } else {
            // the grad of broadcast is reduce
            sanm_assert(ictx[i]->shape.is_batched_scalar());
            TensorND k;
            if (gi.type() == StSparseLinearTrans::ELEMWISE) {
                k = gi.coeff();
                k.reshape_inplace(k.shape().add_axis(k.rank()));
            } else {
                auto gfull = gi.as_full();
                k = gfull.reduce_sum(gfull.rank() - 1, true);
            }
            ictx[i]->jacobian.accum_full(octx.jacobian.is_batched(), k);
        }
    }
}

void MultiplyOprMeta::compute_order_bias(OperatorNode* opr,
                                         ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    const auto& i0ctx = ctx.get(opr->input(0));
    const auto& i1ctx = ctx.get(opr->input(1));
    TensorND& self_bias = octx.get_user_data_or_create<Userdata>().self_bias;

    self_bias.clear();
    for (size_t i = 1; i < ctx.order(); ++i) {
        self_bias.accum_mul(i0ctx.coeffs[i], i1ctx.coeffs[ctx.order() - i]);
    }

    octx.cur_order_bias = self_bias;
    octx.cur_order_bias.accum_mul(i0ctx.coeffs[0], i1ctx.cur_order_bias);
    octx.cur_order_bias.accum_mul(i0ctx.cur_order_bias, i1ctx.coeffs[0]);
}

void MultiplyOprMeta::compute_coeff(OperatorNode* opr,
                                    ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    const auto& i0ctx = ctx.get(opr->input(0));
    const auto& i1ctx = ctx.get(opr->input(1));

    TensorND b = octx.get_user_data_or_create<Userdata>().self_bias;
    b.accum_mul(i0ctx.coeffs[0], i1ctx.coeffs.back());
    b.accum_mul(i0ctx.coeffs.back(), i1ctx.coeffs[0]);
    octx.coeffs.emplace_back(b);
}

const MultiplyOprMeta* MultiplyOprMeta::instance() {
    static MultiplyOprMeta inst;
    return &inst;
}

VarNode* MultiplyOprMeta::make(VarNode* x, VarNode* y) {
    return x->owner_graph()->insert_opr(instance(), nullptr, {x, y})->output(0);
}
