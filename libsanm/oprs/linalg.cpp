/**
 * \file libsanm/oprs/linalg.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/oprs/linalg.h"
#include <cstring>
#include "libsanm/tensor_svd.h"

using namespace sanm;
using namespace symbolic;

namespace {

constexpr size_t conv_size_begin(size_t order, size_t ysize) {
    return order >= ysize ? order - ysize + 1 : 0;
}

constexpr size_t conv_size_end(size_t order, size_t xsize) {
    return std::min(xsize, order + 1);
}

//! convolution of non-constant terms to compute the bias term in the result
void batch_mm_convolution(TensorND& dst, const TensorArray& x,
                          const TensorArray& y, bool trans_x = false,
                          bool trans_y = false, size_t order = 0) {
    dst.clear();
    if (!order) {
        order = x.size();
        sanm_assert(order == y.size());
    }
    for (size_t i = conv_size_begin(order, y.size());
         i < conv_size_end(order, x.size()); ++i) {
        dst.as_batched_mm(x[i], y[order - i], !dst.empty(), trans_x, trans_y);
    }
    if (dst.empty()) {
        dst.set_shape({x[0].shape(0), x[0].shape(1), y[0].shape(2)})
                .fill_with_inplace(0);
    }
}

void batch_mm_convolution_arr(TensorArray& dst, size_t order,
                              const TensorArray& x, const TensorArray& y,
                              bool trans_y = false, bool y_as_diag = false) {
    dst.resize(order + 1);
    for (size_t i = 0; i <= order; ++i) {
        dst[i].clear();
        for (size_t j = conv_size_begin(i, y.size());
             j < conv_size_end(i, x.size()); ++j) {
            if (y_as_diag) {
                auto& yi = y[i - j];
                sanm_assert(yi.rank() == 2);
                dst[i].accum_mul(x[j],
                                 yi.reshape({yi.shape(0), 1, yi.shape(1)}));
            } else {
                dst[i].as_batched_mm(x[j], y[i - j], !dst[i].empty(), false,
                                     trans_y);
            }
        }
        sanm_assert(!dst[i].empty());
    }
}
}  // anonymous namespace

/* ======================= BatchMatInvMulOprMeta ======================= */

void BatchMatInvMulOprMeta::infer_shape(OperatorNode* opr,
                                        ExecutionContext& ctx) const {
    const TensorShape& ishp = ctx.get(opr->input(0)).shape;
    sanm_assert(ishp.rank == 3 && ishp[1] == ishp[2],
                "invalid shape for matinv: %s", ishp.str().c_str());
    if (!param(opr)->use_identity) {
        sanm_assert(ishp == ctx.get(opr->input(1)).shape);
    }
    ctx.get(opr->output(0)).shape = ishp;
}

void BatchMatInvMulOprMeta::eval_bias(OperatorNode* opr,
                                      ExecutionContext& ctx) const {
    const VarNodeExeCtx& x = ctx.get(opr->input(0));
    VarNodeExeCtx& out = ctx.get(opr->output(0));
    sanm_assert(x.coeffs.size() == 1 && out.coeffs.size() == 0);
    auto param = this->param(opr);
    auto& user_data = out.create_user_data<UserData>();
    user_data.xinv = x.coeffs.back().batched_matinv();
    if (param->use_identity) {
        out.coeffs.emplace_back(user_data.xinv);
        return;
    }

    const TensorND& a = ctx.get(opr->input(1)).coeffs.back();
    TensorND& y = out.coeffs.emplace_back();
    if (param->is_left) {
        y.as_batched_mm(a, user_data.xinv);
    } else {
        y.as_batched_mm(user_data.xinv, a);
    }
}

void BatchMatInvMulOprMeta::accum_inp_grad(OperatorNode* opr,
                                           ExecutionContext& ctx) const {
    auto param = this->param(opr);
    VarNodeExeCtx& x = ctx.get(opr->input(0));
    const VarNodeExeCtx& out = ctx.get(opr->output(0));
    auto& user_data = out.get_user_data<UserData>();
    TensorND m0, m1;
    if (param->is_left) {
        m0 = -out.coeffs[0];
        m1 = user_data.xinv;
    } else {
        m0 = user_data.xinv;
        m1 = -out.coeffs[0];
    }

    // gx[b, r, (i, j)] = gy[b, r, (p, q)] * m0[b, p, i] * m1[b, j, q]
    const size_t batch = m0.shape(0), dim = m0.shape(1),
                 graph_odim = out.jacobian.out_dim();
    // out_grad[b, (r, p), q]
    TensorND out_grad = out.jacobian.check_batched(true).as_full().reshape(
            {batch, graph_odim * dim, dim});
    TensorND tmp, gx;

    // tmp[b, (r, i), q] = gy[b, r, (p, q)] * m0[b, p, i]
    tmp.as_batched_mm_vecitem_left(out_grad, m0);

    // gx[b, (r, i), j]
    gx.as_batched_mm(tmp, m1, false, false, true);
    // gx[b, r, (i, j)]
    gx = gx.reshape({batch, graph_odim, dim * dim});

    x.jacobian.accum_full(true, gx);

    if (!param->use_identity) {
        if (param->is_left) {
            // ga[b, r, (i, j)] = gy[b, r, (i, q)] * xinv[b, j, q]
            tmp.as_batched_mm(out_grad, user_data.xinv, false, false, true);
        } else {
            // ga[b, r, (i, j)] = gy[b, r, (p, j)] * xinv[b, p, i]
            tmp.as_batched_mm_vecitem_left(out_grad, user_data.xinv);
        }
        tmp = tmp.reshape({batch, graph_odim, dim * dim});
        ctx.get(opr->input(1)).jacobian.accum_full(true, tmp);
    }
}

void BatchMatInvMulOprMeta::compute_order_bias(OperatorNode* opr,
                                               ExecutionContext& ctx) const {
    auto param = this->param(opr);
    const VarNodeExeCtx& x = ctx.get(opr->input(0));
    VarNodeExeCtx& out = ctx.get(opr->output(0));
    auto& user_data = out.get_user_data<UserData>();
    if (param->is_left) {
        batch_mm_convolution(user_data.self_bias, out.coeffs, x.coeffs);
    } else {
        batch_mm_convolution(user_data.self_bias, x.coeffs, out.coeffs);
    }
    user_data.self_bias.inplace_neg();
    compute_bias(opr, ctx, false);
}

void BatchMatInvMulOprMeta::compute_bias(OperatorNode* opr,
                                         ExecutionContext& ctx,
                                         bool in_coeff) const {
    auto param = this->param(opr);
    const VarNodeExeCtx& x = ctx.get(opr->input(0));
    VarNodeExeCtx& out = ctx.get(opr->output(0));
    const UserData& user_data = out.get_user_data<UserData>();
    TensorND tmp0;
    if (param->use_identity) {
        tmp0 = user_data.self_bias;
    } else {
        tmp0 = ctx.get(opr->input(1)).get_bias(in_coeff) + user_data.self_bias;
    }
    TensorND tmp1;
    if (param->is_left) {
        tmp1.as_batched_mm(out.coeffs[0], x.get_bias(in_coeff));
    } else {
        tmp1.as_batched_mm(x.get_bias(in_coeff), out.coeffs[0]);
    }

    // tmp1: a_k + self_bias - x_kf_0
    tmp1.as_elem<'-'>(tmp0, tmp1);

    TensorND& dst = out.get_bias(in_coeff);
    if (param->is_left) {
        dst.as_batched_mm(tmp1, user_data.xinv);
    } else {
        dst.as_batched_mm(user_data.xinv, tmp1);
    }
}

void BatchMatInvMulOprMeta::compute_coeff(OperatorNode* opr,
                                          ExecutionContext& ctx) const {
    VarNodeExeCtx& out = ctx.get(opr->output(0));
    out.coeffs.emplace_back();
    compute_bias(opr, ctx, true);
}

const BatchMatInvMulOprMeta* BatchMatInvMulOprMeta::instance() {
    static BatchMatInvMulOprMeta inst;
    return &inst;
}

VarNode* BatchMatInvMulOprMeta::make(VarNode* x, VarNode* a, bool is_left) {
    std::unique_ptr<Param> p{new Param};
    p->is_left = is_left;
    VarNodeArray inp{x};
    if (a) {
        inp.push_back(a);
        p->use_identity = false;
    } else {
        p->use_identity = true;
    }
    auto opr = x->owner_graph()->insert_opr(instance(), p.get(), inp);
    p.release();
    return opr->output(0);
}

/* ======================= BatchDeterminantOprMeta ======================= */

void BatchDeterminantOprMeta::infer_shape(OperatorNode* opr,
                                          ExecutionContext& ctx) const {
    const TensorShape& ishp = ctx.get(opr->input(0)).shape;
    sanm_assert(ishp.rank == 3 && ishp[1] == ishp[2],
                "invalid shape for matinv: %s", ishp.str().c_str());
    sanm_assert(ishp[1] > 1, "scalar determinant is unsupported");
    ctx.get(opr->output(0)).shape = {ishp[0], 1};
}

void BatchDeterminantOprMeta::eval_bias(OperatorNode* opr,
                                        ExecutionContext& ctx) const {
    const VarNodeExeCtx& x = ctx.get(opr->input(0));
    VarNodeExeCtx& out = ctx.get(opr->output(0));
    sanm_assert(x.coeffs.size() == 1 && out.coeffs.size() == 0);
    out.coeffs.emplace_back(x.coeffs[0].batched_determinant());
}

void BatchDeterminantOprMeta::accum_inp_grad(OperatorNode* opr,
                                             ExecutionContext& ctx) const {
    VarNodeExeCtx& ictx = ctx.get(opr->input(0));
    VarNodeExeCtx& octx = ctx.get(opr->output(0));
    auto& user_data = octx.create_user_data<UserData>();
    TensorND cofactor = ictx.coeffs[0].batched_cofactor();
    const size_t batch = ictx.shape[0], dim = ictx.shape[1];
    user_data.cofactor_mmreduce = cofactor.reshape({batch, dim * dim, 1});
    ictx.jacobian += octx.jacobian.check_batched(true).compose_with_full(
            cofactor.reshape({batch, 1, dim * dim}));
}

void BatchDeterminantOprMeta::compute_order_bias(OperatorNode* opr,
                                                 ExecutionContext& ctx) const {
    const VarNodeExeCtx& ictx = ctx.get(opr->input(0));
    VarNodeExeCtx& octx = ctx.get(opr->output(0));
    auto& user_data = octx.get_user_data<UserData>();
    user_data.self_bias = compute_polymat_det_coeff(ictx.coeffs, ctx.order());
    const size_t batch = ictx.shape[0], dim = ictx.shape[1];
    octx.cur_order_bias
            .as_batched_mm(ictx.cur_order_bias.reshape({batch, 1, dim * dim}),
                           user_data.cofactor_mmreduce)
            .reshape_inplace({batch, 1}) += user_data.self_bias;
}

void BatchDeterminantOprMeta::compute_coeff(OperatorNode* opr,
                                            ExecutionContext& ctx) const {
    const VarNodeExeCtx& ictx = ctx.get(opr->input(0));
    VarNodeExeCtx& octx = ctx.get(opr->output(0));
    const UserData& user_data = octx.get_user_data<UserData>();
    const size_t batch = ictx.shape[0], dim = ictx.shape[1];
    octx.coeffs.emplace_back()
            .as_batched_mm(ictx.coeffs.back().reshape({batch, 1, dim * dim}),
                           user_data.cofactor_mmreduce)
            .reshape_inplace({batch, 1}) += user_data.self_bias;
}

const BatchDeterminantOprMeta* BatchDeterminantOprMeta::instance() {
    static BatchDeterminantOprMeta inst;
    return &inst;
}

VarNode* BatchDeterminantOprMeta::make(VarNode* x) {
    return x->owner_graph()->insert_opr(instance(), nullptr, {x})->output(0);
}

/* ======================= BatchMatTransposeOprMeta ======================= */

void BatchMatTransposeOprMeta::infer_shape(OperatorNode* opr,
                                           ExecutionContext& ctx) const {
    TensorShape shp = ctx.get(opr->input(0)).shape;
    sanm_assert(shp.rank == 3);
    std::swap(shp.dim[1], shp.dim[2]);
    ctx.get(opr->output(0)).shape = shp;
}

void BatchMatTransposeOprMeta::eval_bias(OperatorNode* opr,
                                         ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    octx.coeffs.emplace_back(ictx.coeffs[0].batched_transpose());
}

void BatchMatTransposeOprMeta::accum_inp_grad(OperatorNode* opr,
                                              ExecutionContext& ctx) const {
    auto& ictx = ctx.get(opr->input(0));
    const auto& octx = ctx.get(opr->output(0));
    auto k = octx.jacobian.check_batched(true).as_full();
    const size_t batch = ictx.shape[0], dim0 = ictx.shape[1],
                 dim1 = ictx.shape[2], odim = octx.jacobian.out_dim();
    k = k.reshape_inplace({batch * odim, dim1, dim0})
                .batched_transpose()
                .reshape_inplace({batch, odim, dim0 * dim1});
    ictx.jacobian.accum_full(true, k);
}

void BatchMatTransposeOprMeta::compute_order_bias(OperatorNode* opr,
                                                  ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    octx.cur_order_bias.as_batched_transpose(ictx.cur_order_bias);
}

void BatchMatTransposeOprMeta::compute_coeff(OperatorNode* opr,
                                             ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    octx.coeffs.emplace_back(ictx.coeffs.back().batched_transpose());
}

const BatchMatTransposeOprMeta* BatchMatTransposeOprMeta::instance() {
    static BatchMatTransposeOprMeta inst;
    return &inst;
}

VarNode* BatchMatTransposeOprMeta::make(VarNode* x) {
    return x->owner_graph()->insert_opr(instance(), nullptr, {x})->output(0);
}

/* ======================= BatchMatMulOprMeta ======================= */

void BatchMatMulOprMeta::infer_shape(OperatorNode* opr,
                                     ExecutionContext& ctx) const {
    const TensorShape& sl = ctx.get(opr->input(0)).shape;
    const TensorShape& sr = ctx.get(opr->input(1)).shape;
    sanm_assert(
            sl.rank == 3 && sr.rank == 3 && sl[0] == sr[0] && sl[2] == sr[1],
            "invalid operand shapes for matmul: %s and %s", sl.str().c_str(),
            sr.str().c_str());
    ctx.get(opr->output(0)).shape = {sl[0], sl[1], sr[2]};
}

void BatchMatMulOprMeta::eval_bias(OperatorNode* opr,
                                   ExecutionContext& ctx) const {
    const auto &ictx0 = ctx.get(opr->input(0)), &ictx1 = ctx.get(opr->input(1));
    auto& octx = ctx.get(opr->output(0));
    sanm_assert(octx.coeffs.empty());
    octx.coeffs.emplace_back();
    octx.coeffs.back().as_batched_mm(ictx0.coeffs[0], ictx1.coeffs[0]);
}

void BatchMatMulOprMeta::accum_inp_grad(OperatorNode* opr,
                                        ExecutionContext& ctx) const {
    auto& i0ctx = ctx.get(opr->input(0));
    auto& i1ctx = ctx.get(opr->input(1));
    const auto& octx = ctx.get(opr->output(0));
    const size_t batch = i0ctx.shape[0], m = i0ctx.shape[1], k = i0ctx.shape[2],
                 n = i1ctx.shape[2], graph_odim = octx.jacobian.out_dim();

    auto gout = octx.jacobian.check_batched(true).as_full().reshape(
            {batch, graph_odim * m, n});

    TensorND g;
    // g0[b, r, (m, k)] = gout[b, r, (m, n)] * i1[b, k, n]
    g.as_batched_mm(gout, i1ctx.coeffs[0], false, false, true)
            .reshape_inplace({batch, graph_odim, m * k});
    i0ctx.jacobian.accum_full(true, g);

    // g1[b, r, (k, n)] = gout[b, r, (m, n)] * i0[b, m, k]
    g.as_batched_mm_vecitem_left(gout, i0ctx.coeffs[0])
            .reshape_inplace({batch, graph_odim, k * n});
    i1ctx.jacobian.accum_full(true, g);
}

void BatchMatMulOprMeta::compute_order_bias(OperatorNode* opr,
                                            ExecutionContext& ctx) const {
    const auto& i0ctx = ctx.get(opr->input(0));
    const auto& i1ctx = ctx.get(opr->input(1));
    auto& octx = ctx.get(opr->output(0));
    auto& user_data = octx.get_user_data_or_create<UserData>();
    batch_mm_convolution(user_data.self_bias, i0ctx.coeffs, i1ctx.coeffs);
    compute_bias(opr, ctx, false);
}

void BatchMatMulOprMeta::compute_bias(OperatorNode* opr, ExecutionContext& ctx,
                                      bool in_coeff) const {
    const auto& i0ctx = ctx.get(opr->input(0));
    const auto& i1ctx = ctx.get(opr->input(1));
    auto& octx = ctx.get(opr->output(0));
    const UserData& user_data = octx.get_user_data<UserData>();
    TensorND& dst = octx.get_bias(in_coeff);
    dst.as_batched_mm(i0ctx.get_bias(in_coeff), i1ctx.coeffs[0]);
    dst.as_batched_mm(i0ctx.coeffs[0], i1ctx.get_bias(in_coeff), true);
    dst += user_data.self_bias;
}

void BatchMatMulOprMeta::compute_coeff(OperatorNode* opr,
                                       ExecutionContext& ctx) const {
    auto& octx = ctx.get(opr->output(0));
    octx.coeffs.emplace_back();
    compute_bias(opr, ctx, true);
}

const BatchMatMulOprMeta* BatchMatMulOprMeta::instance() {
    static BatchMatMulOprMeta inst;
    return &inst;
}

VarNode* BatchMatMulOprMeta::make(VarNode* x, VarNode* y) {
    return x->owner_graph()->insert_opr(instance(), nullptr, {x, y})->output(0);
}

/* ======================= BatchMulEyeOprMeta ======================= */

void BatchMulEyeOprMeta::infer_shape(OperatorNode* opr,
                                     ExecutionContext& ctx) const {
    auto param = this->param(opr);
    TensorShape shp = ctx.get(opr->input(0)).shape;
    sanm_assert(shp.is_batched_scalar(),
                "the input shape must be a scalar: got %s", shp.str().c_str());
    shp.rank = 3;
    shp.dim[1] = shp.dim[2] = param->dim;
    ctx.get(opr->output(0)).shape = shp;
}

void BatchMulEyeOprMeta::eval_bias(OperatorNode* opr,
                                   ExecutionContext& ctx) const {
    const TensorND& ival = ctx.get(opr->input(0)).coeffs[0];
    TensorND& oval = ctx.get(opr->output(0)).coeffs.emplace_back();
    oval.as_batched_diag_scalar(ival, param(opr)->dim);
}

void BatchMulEyeOprMeta::accum_inp_grad(OperatorNode* opr,
                                        ExecutionContext& ctx) const {
    auto& ictx = ctx.get(opr->input(0));
    const auto& octx = ctx.get(opr->output(0));
    TensorND gx, gy = octx.jacobian.check_batched(true).as_full();
    const size_t batch = octx.shape[0], dim = octx.shape[1],
                 odim = octx.jacobian.out_dim();
    gx.as_batched_trace(gy.reshape({batch * odim, dim, dim}))
            .reshape_inplace({batch, odim, 1});
    ictx.jacobian.accum_full(true, gx);
}

void BatchMulEyeOprMeta::compute_order_bias(OperatorNode* opr,
                                            ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    octx.cur_order_bias.as_batched_diag_scalar(ictx.cur_order_bias,
                                               param(opr)->dim);
}

void BatchMulEyeOprMeta::compute_coeff(OperatorNode* opr,
                                       ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    octx.coeffs.emplace_back().as_batched_diag_scalar(ictx.coeffs.back(),
                                                      param(opr)->dim);
}

const BatchMulEyeOprMeta* BatchMulEyeOprMeta::instance() {
    static BatchMulEyeOprMeta inst;
    return &inst;
}

VarNode* BatchMulEyeOprMeta::make(VarNode* x, size_t dim) {
    sanm_assert(dim > 0);
    std::unique_ptr<Param> param{new Param{dim}};
    auto opr = x->owner_graph()->insert_opr(instance(), param.get(), {x});
    param.release();
    return opr->output(0);
}

/* ======================= BatchSVDWOprMeta ======================= */

void BatchSVDWOprMeta::infer_shape(OperatorNode* opr,
                                   ExecutionContext& ctx) const {
    const TensorShape& ishp = ctx.get(opr->input(0)).shape;
    sanm_assert(ishp.rank == 3 && ishp[1] == ishp[2],
                "invalid shape for SVD-W: %s", ishp.str().c_str());
    ctx.get(opr->output(0)).shape = ishp;
    ctx.get(opr->output(1)).shape = {ishp[0], ishp[1]};
    ctx.get(opr->output(2)).shape = ishp;
}

void BatchSVDWOprMeta::eval_bias(OperatorNode* opr,
                                 ExecutionContext& ctx) const {
    const VarNodeExeCtx& ictx = ctx.get(opr->input(0));
    VarNodeExeCtx &uctx = ctx.get(opr->output(0)),
                  &sctx = ctx.get(opr->output(1)),
                  &wctx = ctx.get(opr->output(2));
    ictx.coeffs[0].compute_batched_svd_w(
            uctx.coeffs.emplace_back(), sctx.coeffs.emplace_back(),
            wctx.coeffs.emplace_back(), param(opr)->require_rotation);
}

void BatchSVDWOprMeta::accum_inp_grad(OperatorNode* opr,
                                      ExecutionContext& ctx) const {
    VarNodeExeCtx &ictx = ctx.get(opr->input(0)),
                  &uctx = ctx.get(opr->output(0)),
                  &sctx = ctx.get(opr->output(1)),
                  &wctx = ctx.get(opr->output(2));
    svd_w_grad_revmode(ictx.jacobian, uctx.coeffs[0], sctx.coeffs[0],
                       wctx.coeffs[0], uctx.jacobian, sctx.jacobian,
                       wctx.jacobian);
    wctx.create_user_data<UserData>();
}

void BatchSVDWOprMeta::compute_order_bias(OperatorNode* opr,
                                          ExecutionContext& ctx) const {
    const VarNodeExeCtx& ictx = ctx.get(opr->input(0));
    VarNodeExeCtx &uctx = ctx.get(opr->output(0)),
                  &sctx = ctx.get(opr->output(1)),
                  &wctx = ctx.get(opr->output(2));
    UserData& user_data = wctx.get_user_data<UserData>();
    user_data.mBu.clear();
    user_data.mBw.clear();
    user_data.mMbiask.clear();
    user_data.mBm.clear();
    user_data.mBp.clear();
    user_data.mBpw.clear();
    if (ctx.order() == 1) {
        user_data.mBu = user_data.mBw = user_data.mMbiask = user_data.mBm =
                user_data.mBp = user_data.mBpw = ictx.coeffs[0].fill_with(0);
        sanm_assert(user_data.P.empty());
        user_data.pw_mode = !uctx.nr_reader() && !sctx.nr_reader();
        if (user_data.pw_mode) {
            user_data.P.emplace_back();  // P0 is not used
            wctx.cur_order_bias = wctx.coeffs[0].fill_with(0);
        } else {
            for (auto i : {&uctx, &sctx, &wctx}) {
                i->cur_order_bias = i->coeffs[0].fill_with(0);
            }
        }
        return;
    }
    if (user_data.pw_mode) {
        sanm_assert(user_data.P.size() == ctx.order());
        batch_mm_convolution(user_data.mBm, ictx.coeffs, ictx.coeffs, false,
                             true);
        batch_mm_convolution(user_data.mBp, user_data.P, user_data.P, true,
                             false);
        batch_mm_convolution(user_data.mBpw, user_data.P, wctx.coeffs);
    } else {
        batch_mm_convolution(user_data.mBu, uctx.coeffs, uctx.coeffs, true);
        batch_mm_convolution(user_data.mBw, wctx.coeffs, wctx.coeffs, true);
        TensorArray tmp0, tmp1;
        // US
        batch_mm_convolution_arr(tmp0, ctx.order(), uctx.coeffs, sctx.coeffs,
                                 false, true);
        // USU'
        batch_mm_convolution_arr(tmp1, ctx.order(), tmp0, uctx.coeffs, true);
        // USU'W
        batch_mm_convolution(user_data.mMbiask, tmp1, wctx.coeffs, false, false,
                             ctx.order());
    }

    compute_bias(opr, ctx, false);
}

void BatchSVDWOprMeta::compute_bias(OperatorNode* opr, ExecutionContext& ctx,
                                    bool in_coeff) const {
    const VarNodeExeCtx& ictx = ctx.get(opr->input(0));
    VarNodeExeCtx &uctx = ctx.get(opr->output(0)),
                  &sctx = ctx.get(opr->output(1)),
                  &wctx = ctx.get(opr->output(2));
    UserData& user_data = wctx.get_user_data<UserData>();
    if (user_data.pw_mode) {
        TensorND mPk;
        svd_w_taylor_fwd_p(mPk, wctx.get_bias(in_coeff),
                           ictx.get_bias(in_coeff), uctx.coeffs[0],
                           sctx.coeffs[0], wctx.coeffs[0], user_data.mBm,
                           user_data.mBp, user_data.mBpw);
        if (in_coeff) {
            user_data.P.emplace_back(mPk);
        }
    } else {
        svd_w_taylor_fwd(uctx.get_bias(in_coeff), sctx.get_bias(in_coeff),
                         wctx.get_bias(in_coeff), ictx.get_bias(in_coeff),
                         user_data.mMbiask, uctx.coeffs[0], sctx.coeffs[0],
                         wctx.coeffs[0], &user_data.mBu, user_data.mBw);
    }
}

void BatchSVDWOprMeta::compute_coeff(OperatorNode* opr,
                                     ExecutionContext& ctx) const {
    VarNodeExeCtx &uctx = ctx.get(opr->output(0)),
                  &sctx = ctx.get(opr->output(1)),
                  &wctx = ctx.get(opr->output(2));
    if (!ctx.get(opr->output(2)).get_user_data<UserData>().pw_mode) {
        uctx.coeffs.emplace_back();
        sctx.coeffs.emplace_back();
    }
    wctx.coeffs.emplace_back();
    compute_bias(opr, ctx, true);
}

const BatchSVDWOprMeta* BatchSVDWOprMeta::instance() {
    static BatchSVDWOprMeta inst;
    return &inst;
}

OperatorNode* BatchSVDWOprMeta::make(VarNode* x, bool require_rotation) {
    std::unique_ptr<Param> p{new Param{require_rotation}};
    auto opr = x->owner_graph()->insert_opr(instance(), p.get(), {x});
    p.release();
    return opr;
}
