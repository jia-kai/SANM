/**
 * \file libsanm/tensor_svd.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once
#include "libsanm/tensor.h"

namespace sanm {
/*!
 * \brief reverse mode gradient propagation for SVD
 *
 * mM is the original input matrix, with mU, mS and mW its SVD, and mdU, mdS and
 * mdW the corresponding Jacobians of the output var.
 */
void svd_w_grad_revmode(StSparseLinearTrans& grad, const TensorND& mU,
                        const TensorND& mS, const TensorND& mW,
                        const StSparseLinearTrans& mdU,
                        const StSparseLinearTrans& mdS,
                        const StSparseLinearTrans& mdW);

/*!
 * \brief propagating the current highest order taylor expansion coefficient
 *
 * \param mBw bias of solved terms in the expansion of W'W = I
 * \param mBu see \p mBw. If mBu is null, then \p mUs and \p mSk will not be
 *      computed
 *
 * Note that \p mMk should include the bias term (sum U_iS_jU_k^TW_l such that
 * max(i, j, k, l) < order). Both \p mBu and \p mBw should be symmetric.
 */
void svd_w_taylor_fwd(TensorND& mUk, TensorND& mSk, TensorND& mWk,
                      const TensorND& mMk, const TensorND& mMbiask,
                      const TensorND& mU0, const TensorND& mS0,
                      const TensorND& mW0, const TensorND* mBu,
                      const TensorND& mBw);

/*!
 * \brief forward in the M=PW mode where P=USU', when U and S are not needed in
 *      SVD-W
 *
 * This is in fact the polar decomposition.
 *
 * \param mBm bias in the expansion MM' (i.e., MiMj' where i+j=k, i<k, j<k)
 * \param mBp bias in the expansion P'P
 * \param mBpw bias in the expansion PW
 */
void svd_w_taylor_fwd_p(TensorND& mPk, TensorND& mWk, const TensorND& mMk,
                        const TensorND& mU0, const TensorND& mS0,
                        const TensorND& mW0, const TensorND& mBm,
                        const TensorND& mBp, const TensorND& mBpw);

}  // namespace sanm
