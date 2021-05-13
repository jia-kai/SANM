/**
 * \file libsanm/oprs/misc.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/oprs/misc.h"

using namespace sanm;
using namespace symbolic;

/* ======================= PlaceholderOprMeta ======================= */

void PlaceholderOprMeta::infer_shape(OperatorNode* opr,
                                     ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    sanm_assert(octx.coeffs.size() == 1);
    octx.shape = octx.coeffs[0].shape();
}

void PlaceholderOprMeta::eval_bias(OperatorNode* opr,
                                   ExecutionContext& ctx) const {
    sanm_assert(ctx.get(opr->output(0)).coeffs.size() == 1,
                "output of placeholder should have been initialized");
}

void PlaceholderOprMeta::accum_inp_grad(OperatorNode*,
                                        ExecutionContext&) const {}

void PlaceholderOprMeta::compute_order_bias(OperatorNode* opr,
                                            ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    octx.cur_order_bias = octx.coeffs[0].fill_with(0);
}

void PlaceholderOprMeta::compute_coeff(OperatorNode* opr,
                                       ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    sanm_assert(octx.coeffs.size() == ctx.order() + 1);
}

const PlaceholderOprMeta* PlaceholderOprMeta::instance() {
    static PlaceholderOprMeta inst;
    return &inst;
}

/* ======================= ConstantOprMeta ======================= */

void ConstantOprMeta::infer_shape(OperatorNode* opr,
                                  ExecutionContext& ctx) const {
    TensorShape shp = param(opr)->val.shape();
    if (auto& shard = ctx.parallel_shard(); shard.valid()) {
        if (shp.rank && shp[0] > 1) {
            sanm_assert(shp[0] == shard->tot,
                        "ConstantOprMeta shape mismatch in data parallel: "
                        "tot_batch=%zu value_shape=%zu",
                        shard->tot, shp[0]);
            shp.dim[0] = shard->end - shard->begin;
        }
    }
    ctx.get(opr->output(0)).shape = shp;
}

void ConstantOprMeta::eval_bias(OperatorNode* opr,
                                ExecutionContext& ctx) const {
    const TensorND& src = param(opr)->val;
    TensorND& dst = ctx.get(opr->output(0)).coeffs.emplace_back();
    if (auto& shard = ctx.parallel_shard(); shard.valid() && src.shape(0) > 1) {
        dst.copy_from_sub_batch(src, 0, shard->begin,
                                shard->end - shard->begin);
    } else {
        dst = src;
    }
}

void ConstantOprMeta::accum_inp_grad(OperatorNode*, ExecutionContext&) const {}

void ConstantOprMeta::compute_order_bias(OperatorNode* opr,
                                         ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    octx.cur_order_bias = octx.coeffs[0].fill_with(0);
}

void ConstantOprMeta::compute_coeff(OperatorNode* opr,
                                    ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    octx.coeffs.emplace_back(octx.coeffs[0].fill_with(0));
}

const ConstantOprMeta* ConstantOprMeta::instance() {
    static ConstantOprMeta inst;
    return &inst;
}

VarNode* ConstantOprMeta::make(ComputingGraph& cg, TensorND val) {
    sanm_assert(!val.empty());
    std::unique_ptr<Param> param{new Param{std::move(val)}};
    auto opr = cg.insert_opr(instance(), param.get(), {});
    param.release();
    return opr->output(0);
}

/* ======================= SliceOprMeta ======================= */

std::pair<int, int> SliceOprMeta::abs_interval(OperatorNode* opr,
                                               int size) const {
    auto param = this->param(opr);
    sanm_assert(param->stride != 0 && size > 0);
    int begin, end;
    if (param->begin.valid()) {
        begin = param->begin.val();
        if (begin < 0) {
            begin += size;
        }
    } else {
        begin = param->stride > 0 ? 0 : size - 1;
    }
    if (param->end.valid()) {
        end = param->end.val();
        if (end < 0) {
            end += size;
        }
    } else {
        end = param->stride > 0 ? size : -1;
    }
    if (param->stride < 0) {
        sanm_assert(begin > end && end >= -1 && begin < size);
    } else {
        sanm_assert(begin < end && begin >= 0 && end <= size);
    }
    return {begin, end};
}

void SliceOprMeta::infer_shape(OperatorNode* opr, ExecutionContext& ctx) const {
    auto param = this->param(opr);
    TensorShape oshp = ctx.get(opr->input(0)).shape;
    sanm_assert(param->axis >= 0 &&
                static_cast<size_t>(param->axis) < oshp.rank);
    int begin, end;
    std::tie(begin, end) = abs_interval(opr, oshp[param->axis]);
    oshp.dim[param->axis] =
            (std::abs(end - begin) - 1) / std::abs(param->stride) + 1;
    ctx.get(opr->output(0)).shape = oshp;
}

void SliceOprMeta::compute(OperatorNode* opr, TensorND& dst,
                           const TensorND& src) const {
    auto param = this->param(opr);
    sanm_assert(param->axis == 1, "axis %d unimplemented", param->axis);
    sanm_assert(param->stride == 1, "stride %d unimplemented", param->stride);
    sanm_assert(src.shape(0) == 1, "only support one batch");
    int begin, end;
    std::tie(begin, end) = abs_interval(opr, src.shape(1));
    dst.clear();
    dst.copy_from_sub_batch(src.reshape(src.shape().remove_axis(0)), 0, begin,
                            end - begin);
    dst.reshape_inplace(dst.shape().add_axis(0));
}

void SliceOprMeta::eval_bias(OperatorNode* opr, ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    compute(opr, octx.coeffs.emplace_back(), ictx.coeffs[0]);
}

void SliceOprMeta::accum_inp_grad(OperatorNode* opr,
                                  ExecutionContext& ctx) const {
    auto& ictx = ctx.get(opr->input(0));
    const auto& octx = ctx.get(opr->output(0));
    auto param = this->param(opr);
    sanm_assert(param->axis == 1, "axis %d unimplemented", param->axis);
    sanm_assert(param->stride == 1, "stride %d unimplemented", param->stride);
    sanm_assert(ictx.shape.rank == 2 && ictx.shape[0] == 1,
                "shape %s unimplemented", ictx.shape.str().c_str());

    // XXX; inefficient implementation
    sanm_assert(octx.jacobian.is_batched() && octx.jacobian.batch() == 1 &&
                ictx.shape.rank == 2);
    TensorND j;
    if (ictx.jacobian.valid()) {
        j.as_batched_transpose(ictx.jacobian.check_batched(true).as_full());
        j.reshape_inplace({ictx.shape[1], octx.jacobian.out_dim()});
        ictx.jacobian = {};
    } else {
        j.set_shape({ictx.shape[1], octx.jacobian.out_dim()})
                .fill_with_inplace(0);
    }
    int begin, end;
    std::tie(begin, end) = abs_interval(opr, ictx.shape[1]);
    TensorND jo;
    jo.as_batched_transpose(octx.jacobian.as_full());
    jo.reshape_inplace(jo.shape().remove_axis(0));
    j.incr_sub_batch(jo, begin, 0, end - begin, 1, 1);
    TensorND& jt = jo;  // reuse temp
    jt.as_transpose(j);
    jt.reshape_inplace(jt.shape().add_axis(0));
    ictx.jacobian.reset(StSparseLinearTrans::FULL, true, jt);
}

void SliceOprMeta::compute_order_bias(OperatorNode* opr,
                                      ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    if (ctx.order() == 1) {
        octx.cur_order_bias = octx.coeffs[0].fill_with(0);
    } else {
        compute(opr, octx.cur_order_bias, ictx.cur_order_bias);
    }
}

void SliceOprMeta::compute_coeff(OperatorNode* opr,
                                 ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    compute(opr, octx.coeffs.emplace_back(), ictx.coeffs.back());
}

const SliceOprMeta* SliceOprMeta::instance() {
    static SliceOprMeta inst;
    return &inst;
}

VarNode* SliceOprMeta::make(VarNode* x, int axis, Maybe<int> begin,
                            Maybe<int> end, int stride) {
    sanm_assert(axis >= 0 && stride != 0);
    std::unique_ptr<Param> param{new Param{axis, stride, begin, end}};
    auto opr = x->owner_graph()->insert_opr(instance(), param.get(), {x});
    param.release();
    return opr->output(0);
}

/* ======================= ConcatOprMeta ======================= */
void ConcatOprMeta::infer_shape(OperatorNode* opr,
                                ExecutionContext& ctx) const {
    TensorShape& oshp = ctx.get(opr->output(0)).shape;
    sanm_assert(!oshp.rank);
    auto param = this->param(opr);
    size_t axis = param->axis;
    for (VarNode* i : opr->inputs()) {
        TensorShape ishp = ctx.get(i).shape;
        if (!oshp.rank) {
            oshp = ishp;
        } else {
            sanm_assert(axis <= oshp.rank && ishp.rank == oshp.rank);
            oshp.dim[axis] += ishp.dim[axis];
            ishp.dim[axis] = oshp[axis];
            sanm_assert(ishp == oshp,
                        "concat shape mismatch %s vs %s (axis=%zu)",
                        ishp.str().c_str(), oshp.str().c_str(), axis);
        }
    }
    sanm_assert(oshp.rank);
}

void ConcatOprMeta::eval_bias(OperatorNode* opr, ExecutionContext& ctx) const {
    ctx.get(opr->output(0)).coeffs.emplace_back();
    compute(opr, ctx, true);
}

void ConcatOprMeta::accum_inp_grad(OperatorNode* opr,
                                   ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    sanm_assert(param(opr)->axis == 1 && octx.shape.rank == 2 &&
                        octx.jacobian.is_batched() &&
                        octx.jacobian.batch() == 1,
                "unimplemented");
    TensorND joT;
    joT.as_batched_transpose(octx.jacobian.as_full());
    joT.reshape_inplace({joT.shape(1), joT.shape(2)});
    size_t off = 0;
    TensorND jiT, ji;
    for (VarNode* i : opr->inputs()) {
        auto& ictx = ctx.get(i);
        jiT.clear();
        jiT.copy_from_sub_batch(joT, 0, off, ictx.shape[1]);
        off += ictx.shape[1];
        ji.as_transpose(jiT);
        ji.reshape_inplace(ji.shape().add_axis(0));
        ictx.jacobian.accum_full(true, ji);
    }
    sanm_assert(off == octx.shape[1]);
}

void ConcatOprMeta::compute_order_bias(OperatorNode* opr,
                                       ExecutionContext& ctx) const {
    if (ctx.order() == 1) {
        auto& octx = ctx.get(opr->output(0));
        octx.cur_order_bias = octx.coeffs[0].fill_with(0);
    } else {
        compute(opr, ctx, false);
    }
}

void ConcatOprMeta::compute_coeff(OperatorNode* opr,
                                  ExecutionContext& ctx) const {
    ctx.get(opr->output(0)).coeffs.emplace_back();
    compute(opr, ctx, true);
}

void ConcatOprMeta::compute(OperatorNode* opr, ExecutionContext& ctx,
                            bool in_coeff) const {
    auto& octx = ctx.get(opr->output(0));
    sanm_assert(param(opr)->axis == 1 && octx.shape[0] == 1, "unimplemented");
    TensorND& dst = octx.get_bias(in_coeff);
    dst.set_shape(octx.shape.remove_axis(0));
    size_t off = 0;
    for (VarNode* i : opr->inputs()) {
        auto& ictx = ctx.get(i);
        size_t size = ictx.shape[1];
        dst.copy_from_sub_batch(ictx.get_bias(in_coeff).reshape({size}), off, 0,
                                size);
        off += size;
    }
    sanm_assert(off == octx.shape[1]);
    dst.reshape_inplace(octx.shape);
}

const ConcatOprMeta* ConcatOprMeta::instance() {
    static ConcatOprMeta inst;
    return &inst;
}

VarNode* ConcatOprMeta::make(std::span<VarNode*> inputs, int axis) {
    sanm_assert(!inputs.empty() && axis >= 0);
    std::unique_ptr<Param> param{new Param};
    param->nr_input = inputs.size();
    param->axis = axis;
    auto opr = inputs[0]->owner_graph()->insert_opr(
            instance(), param.get(), {inputs.begin(), inputs.end()});
    param.release();
    return opr->output(0);
}
