/**
 * \file libsanm/oprs/linalg.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// linear algebra operators
#pragma once

#include "libsanm/symbolic.h"

namespace sanm {
namespace symbolic {

//! an opr that computes inv(x)*a (!is_left) or a*inv(x) (is_left)
class BatchMatInvMulOprMeta final : public OperatorMeta {
    struct Param {
        bool use_identity;
        bool is_left;
    };
    struct UserData final : public VarNodeExeCtx::Userdata {
        TensorND xinv, self_bias;
    };

    BatchMatInvMulOprMeta() = default;

    static Param* param(OperatorNode* opr) {
        return static_cast<Param*>(opr->storage());
    }

    void compute_bias(OperatorNode* opr, ExecutionContext& ctx,
                      bool in_coeff) const;

public:
    const char* name() const override { return "batch_matinv"; }

    size_t nr_input(OperatorNode* opr) const override {
        return param(opr)->use_identity ? 1 : 2;
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

    static const BatchMatInvMulOprMeta* instance();

    static VarNode* make(VarNode* x, VarNode* a, bool is_left);
};

class BatchDeterminantOprMeta final : public OperatorMeta {
    struct UserData final : public VarNodeExeCtx::Userdata {
        TensorND cofactor_mmreduce;  //!< (b, m*m, 1) for reducing input coeffs
        TensorND self_bias;
    };

    BatchDeterminantOprMeta() = default;

public:
    const char* name() const override { return "batch_det"; }

    size_t nr_input(OperatorNode* opr) const override { return 1; }

    size_t nr_output(OperatorNode*) const override { return 1; }

    void on_opr_del(OperatorNode*) const noexcept override {}

    void infer_shape(OperatorNode* opr, ExecutionContext& ctx) const override;

    void eval_bias(OperatorNode* opr, ExecutionContext& ctx) const override;

    void accum_inp_grad(OperatorNode* opr,
                        ExecutionContext& ctx) const override;

    void compute_order_bias(OperatorNode* opr,
                            ExecutionContext& ctx) const override;

    void compute_coeff(OperatorNode* opr, ExecutionContext& ctx) const override;

    static const BatchDeterminantOprMeta* instance();

    static VarNode* make(VarNode* x);
};

class BatchMatTransposeOprMeta final : public OperatorMeta {
    BatchMatTransposeOprMeta() = default;

public:
    const char* name() const override { return "batch_transpose"; }

    size_t nr_input(OperatorNode* opr) const override { return 1; }

    size_t nr_output(OperatorNode*) const override { return 1; }

    void on_opr_del(OperatorNode*) const noexcept override {}

    void infer_shape(OperatorNode* opr, ExecutionContext& ctx) const override;

    void eval_bias(OperatorNode* opr, ExecutionContext& ctx) const override;

    void accum_inp_grad(OperatorNode* opr,
                        ExecutionContext& ctx) const override;

    void compute_order_bias(OperatorNode* opr,
                            ExecutionContext& ctx) const override;

    void compute_coeff(OperatorNode* opr, ExecutionContext& ctx) const override;

    static const BatchMatTransposeOprMeta* instance();

    static VarNode* make(VarNode* x);
};

class BatchMatMulOprMeta final : public OperatorMeta {
    struct UserData final : public VarNodeExeCtx::Userdata {
        TensorND self_bias;
    };

    BatchMatMulOprMeta() = default;

    void compute_bias(OperatorNode* opr, ExecutionContext& ctx,
                      bool in_coeff) const;

public:
    const char* name() const override { return "batch_matmul"; }

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

    static const BatchMatMulOprMeta* instance();

    static VarNode* make(VarNode* x, VarNode* y);
};

//! multiply a scalar with the identity matrix
class BatchMulEyeOprMeta final : public OperatorMeta {
    struct Param {
        size_t dim;
    };
    BatchMulEyeOprMeta() = default;

    Param* param(OperatorNode* opr) const {
        sanm_assert(opr->meta() == this);
        return static_cast<Param*>(opr->storage());
    }

public:
    const char* name() const override { return "batch_mul_eye"; }

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

    static const BatchMulEyeOprMeta* instance();

    static VarNode* make(VarNode* x, size_t dim);
};

//! SVD-W; see TensorND::compute_batched_svd_w
class BatchSVDWOprMeta final : public OperatorMeta {
    struct Param {
        bool require_rotation;
    };
    struct UserData final : public VarNodeExeCtx::Userdata {
        bool pw_mode;  //! true if we compute P=US when U and S are not used
        TensorND mBu, mBw, mMbiask, mBm, mBp, mBpw;
        TensorArray P;
    };

    Param* param(OperatorNode* opr) const {
        sanm_assert(opr->meta() == this);
        return static_cast<Param*>(opr->storage());
    }

    void compute_bias(OperatorNode* opr, ExecutionContext& ctx,
                      bool in_coeff) const;

public:
    const char* name() const override { return "batch_svd_w"; }

    size_t nr_input(OperatorNode*) const override { return 1; }

    size_t nr_output(OperatorNode*) const override { return 3; }

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

    static const BatchSVDWOprMeta* instance();

    static OperatorNode* make(VarNode* x, bool require_rotation);
};

}  // namespace symbolic
}  // namespace sanm
