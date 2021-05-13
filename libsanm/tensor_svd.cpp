/**
 * \file libsanm/tensor_svd.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/tensor_svd.h"
#include "libsanm/tensor_impl_helper.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace sanm;

namespace {
void assign_check(size_t& dst, size_t v) {
    if (!dst) {
        dst = v;
    } else {
        sanm_assert(dst == v);
    }
}

fp_t* get_full_ptr(TensorND& dst, const StSparseLinearTrans& lt) {
    dst = lt.check_batched(true).as_full();
    return const_cast<fp_t*>(dst.ptr());
}

fp_t clip_div(fp_t x, fp_t y) {
    constexpr fp_t eps = 1e-12;
    return x * y / (y * y + eps);
}

fp_t* ptr_offset(fp_t* p, size_t off) {
    return p ? p + off : nullptr;
}

#define FOR4_BEGIN(i, j, k, l, n)              \
    do {                                       \
        for (size_t i = 0; i < n; ++i)         \
            for (size_t j = 0; j < n; ++j)     \
                for (size_t k = 0; k < n; ++k) \
                    for (size_t l = 0; l < n; ++l)
#define FOR4_END() \
    }              \
    while (0)
}  // anonymous namespace

const TensorND& TensorND::compute_batched_svd_w(TensorND& u, TensorND& s,
                                                TensorND& w,
                                                bool require_rotation) const {
    SANM_SCOPED_PROFILER("batched_svd_w");
    sanm_assert(rank() == 3);
    const size_t batch = shape(0), n = shape(1);
    sanm_assert(n >= 2);
    sanm_assert(shape(2) == n, "not square matrices: %s",
                shape().str().c_str());
    u.set_shape({batch, n, n});
    s.set_shape({batch, n});
    w.set_shape({batch, n, n});

    auto mptr = ptr();
    auto uptr = u.woptr(), sptr = s.woptr(), wptr = w.woptr();
    auto run = [mptr, uptr, sptr, wptr, batch, n,
                require_rotation]<Eigen::Index sn>() {
        ScopedAllowMalloc scoped_allow_malloc;
        Eigen::JacobiSVD<Eigen::Matrix<fp_t, sn, sn>> svd(
                n, n, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<fp_t, sn, sn> msrcT(n, n);
        // note: with EIGEN_USE_LAPACKE, it JacobiSVD seems to fallback to MKL
        // which would not be so slow
        sanm_assert(n <= 16, "TODO: switch to BDCSVD for large matrices");
        if (sn != Eigen::Dynamic) {
            if (static_cast<Eigen::Index>(n) != sn) {
                // hint the compiler that n equals to sn
                __builtin_unreachable();
            }
        }
        for (size_t i = 0; i < batch; ++i) {
            EigenMat<sn, sn> msrc(const_cast<fp_t*>(mptr) + i * n * n, n, n);
            msrcT = msrc;  // eigen svd class does not take Map inputs
            svd.compute(msrcT);
            EigenMat<sn, 1> ms(sptr + i * n, n, 1);
            EigenMat<sn, sn> muT(uptr + i * n * n, n, n);
            EigenMat<sn, sn> mwT(wptr + i * n * n, n, n);
            // eigen uses column-major and computes M' = VSU'
            ms = svd.singularValues();
            muT = svd.matrixV().transpose();
            if (require_rotation && ((svd.matrixU().determinant() < 0) !=
                                     (svd.matrixV().determinant() < 0))) {
#if 1
                // negate some singular values so that det(w) = 1
                constexpr fp_t EPS = 1e-3;
                int best_idx = -1, best_idx_nr = n + 1;
                for (size_t i = 0; i < n; ++i) {
                    size_t j = i + 1;
                    // ms already sorted
                    while (j < n && std::fabs(ms(i) - ms(j)) < EPS) {
                        ++j;
                    }
                    int nr = j - i;
                    // best case is to negate an odd number of smallest singular
                    // values (so si+sj != 0 in the hessian);
                    // otherwise negate one value whose has the least
                    // repetitionss
                    if (nr <= best_idx_nr ||
                        (nr == best_idx_nr + 1 && nr % 2 == 1)) {
                        best_idx = i;
                        best_idx_nr = nr;
                        if (nr == 1) {
                            break;
                        }
                    }
                    i = j;
                }
                if (best_idx_nr == 1 || best_idx_nr % 2 == 0) {
                    ms(best_idx) = -ms(best_idx);
                    muT.row(best_idx) = -muT.row(best_idx);
                } else {
                    for (int i = best_idx; i < best_idx + best_idx_nr; ++i) {
                        ms(i) = -ms(i);
                        muT.row(i) = -muT.row(i);
                    }
                }
#else
                ms(n - 1) = -ms(n - 1);
                muT.row(n - 1) = -muT.row(n - 1);
#endif
            }
            // w=uv', w'=vu'
            mwT.noalias() = svd.matrixU() * muT;
        }
    };
#define CASE(sn)                  \
    do {                          \
        if (n == sn) {            \
            run.operator()<sn>(); \
            return *this;         \
        }                         \
    } while (0)
    CASE(2);
    CASE(3);
#undef CASE
    run.operator()<Eigen::Dynamic>();
    return *this;
}

void sanm::svd_w_grad_revmode(StSparseLinearTrans& grad, const TensorND& mU,
                              const TensorND& mS, const TensorND& mW,
                              const StSparseLinearTrans& mdU,
                              const StSparseLinearTrans& mdS,
                              const StSparseLinearTrans& mdW) {
    SANM_SCOPED_PROFILER("batched_svd_w_grad");
    auto run = [&]<Eigen::Index sn>() {
        // extract sizes and pointers
        const size_t batch = mU.shape(0), n = mU.shape(1);
        if (sn != Eigen::Dynamic && static_cast<Eigen::Index>(n) != sn) {
            // hint the compiler that n equals to sn
            __builtin_unreachable();
        }
        TensorND mdU_full, mdS_full, mdW_full, mdM;
        fp_t *mdU_ptr = nullptr, *mdS_ptr = nullptr, *mdW_ptr = nullptr;
        size_t out_dim = 0;
        if (mdU.valid()) {
            mdU_ptr = get_full_ptr(mdU_full, mdU);
            assign_check(out_dim, mdU.out_dim());
        }
        if (mdS.valid()) {
            mdS_ptr = get_full_ptr(mdS_full, mdS);
            assign_check(out_dim, mdS.out_dim());
        }
        if (mdW.valid()) {
            mdW_ptr = get_full_ptr(mdW_full, mdW);
            assign_check(out_dim, mdW.out_dim());
        }
        sanm_assert(out_dim, "no output grad");
        if (grad.valid()) {
            mdM = grad.check_batched(true).as_full();
            sanm_assert(mdM.shape() == (TensorShape{batch, out_dim, n * n}));
            grad = {};
        } else {
            mdM.set_shape({batch, out_dim, n * n}).fill_with_inplace(0);
        }
        auto mdMptr = mdM.rwptr();
        auto mU_ptr = const_cast<fp_t*>(mU.ptr()),
             mS_ptr = const_cast<fp_t*>(mS.ptr()),
             mW_ptr = const_cast<fp_t*>(mW.ptr());

        // allocate temporaries
        ScopedAllowMalloc scoped_allow_malloc;
        constexpr Eigen::Index snsqr = sn == Eigen::Dynamic ? sn : sn * sn;
        Eigen::Matrix<fp_t, sn, sn> cV(n, n);
        Eigen::Matrix<fp_t, snsqr, sn> dsdmT(n * n, n);
        Eigen::Matrix<fp_t, snsqr, snsqr> dxdmT(n * n, n * n),
                dydmT(n * n, n * n), dwdyT(n * n, n * n), dudxT(n * n, n * n),
                tmpn2(n * n, n * n);
        scoped_allow_malloc.disallow();

        // matrix variables:
        // prefix c means current (this batch)
        // prefix d means derivative/jacobian
        // suffix t/T means transposed (eigen uses col-major)

        for (size_t ib = 0; ib < batch; ++ib) {
            size_t ib_off = ib * n * n;
            EigenMat<sn, sn> cUt(mU_ptr + ib_off, n, n),
                    cWt(mW_ptr + ib_off, n, n);
            EigenMat<sn, 1> cS(mS_ptr + ib * n, n, 1);
            EigenMat<snsqr, Eigen::Dynamic> cdMt(mdMptr + ib_off * out_dim,
                                                 n * n, out_dim);
            cV.noalias() = cWt * cUt.transpose();
            if (mdS_ptr) {
                for (size_t si = 0; si < n; ++si) {
                    for (size_t mi = 0; mi < n; ++mi) {
                        for (size_t mj = 0; mj < n; ++mj) {
                            dsdmT(mi * n + mj, si) = cUt(si, mi) * cV(mj, si);
                        }
                    }
                }
                EigenMat<sn, Eigen::Dynamic> cdSt(mdS_ptr + ib * n * out_dim, n,
                                                  out_dim);
                cdMt.noalias() += dsdmT * cdSt;
            }
            if (mdW_ptr || mdU_ptr) {
                FOR4_BEGIN(i, j, k, l, n) {
                    fp_t cij = cUt(i, k) * cV(l, j), cji = cUt(j, k) * cV(l, i),
                         si = cS(i), sj = cS(j);
                    if (mdW_ptr) {
                        dydmT(k * n + l, i * n + j) =
                                i == j ? 0 : clip_div(cij - cji, si + sj);
                        dwdyT(k * n + l, i * n + j) = cUt(k, i) * cV(j, l);
                    }
                    if (mdU_ptr) {
                        dudxT(k * n + l, i * n + j) = l == j ? cUt(k, i) : 0;
                        dxdmT(k * n + l, i * n + j) =
                                i == j ? 0
                                       : clip_div(cij * sj + cji * si,
                                                  sj * sj - si * si);
                    }
                }
                FOR4_END();

                if (mdW_ptr) {
                    tmpn2.noalias() = dydmT * dwdyT;
                    EigenMat<snsqr, Eigen::Dynamic> cdWt(
                            mdW_ptr + ib_off * out_dim, n * n, out_dim);
                    cdMt.noalias() += tmpn2 * cdWt;
                }
                if (mdU_ptr) {
                    tmpn2.noalias() = dxdmT * dudxT;
                    EigenMat<snsqr, Eigen::Dynamic> cdUt(
                            mdU_ptr + ib_off * out_dim, n * n, out_dim);
                    cdMt.noalias() += tmpn2 * cdUt;
                }
            }
        }

        grad.reset(StSparseLinearTrans::FULL, true, mdM);
    };

    size_t n = mU.shape(1);
    sanm_assert(n >= 2);
#define CASE(sn)                  \
    do {                          \
        if (n == sn) {            \
            run.operator()<sn>(); \
            return;               \
        }                         \
    } while (0)
    CASE(2);
    CASE(3);
#undef CASE
    run.operator()<Eigen::Dynamic>();
}

void sanm::svd_w_taylor_fwd(TensorND& mUk, TensorND& mSk, TensorND& mWk,
                            const TensorND& mMk, const TensorND& mMbiask,
                            const TensorND& mU0, const TensorND& mS0,
                            const TensorND& mW0, const TensorND* mBu,
                            const TensorND& mBw) {
    SANM_SCOPED_PROFILER("svd_w_taylor_fwd");
    auto run = [&]<Eigen::Index sn>() {
        const size_t batch = mMk.shape(0), n = mMk.shape(1);
        if (sn != Eigen::Dynamic && static_cast<Eigen::Index>(n) != sn) {
            // hint the compiler that n equals to sn
            __builtin_unreachable();
        }
        fp_t *mUkptr = mBu ? mUk.woptr() : nullptr,
             *mSkptr = mBu ? mSk.woptr() : nullptr, *mWkptr = mWk.woptr();
#define DEF(x) auto x##ptr = const_cast<fp_t*>(x.ptr())
        DEF(mMk);
        DEF(mMbiask);
        DEF(mU0);
        DEF(mS0);
        DEF(mW0);
        DEF(mBw);
#undef DEF
        fp_t* mBuptr = mBu ? const_cast<fp_t*>(mBu->ptr()) : nullptr;

        // matrix variables:
        // prefix c means current (this batch)
        // suffix t/T means transposed (eigen uses col-major)

        // allocate temporaries
        ScopedAllowMalloc scoped_allow_malloc;
        Eigen::Matrix<fp_t, sn, sn> cV0(n, n), eqbT(n, n), tmp0(n, n),
                tmp1(n, n);
        MatrixMultiProduct<Eigen::Matrix<fp_t, sn, sn>> mprod(n, n);
        scoped_allow_malloc.disallow();

        for (size_t ib = 0; ib < batch; ++ib) {
            size_t ib_off = ib * n * n;
            EigenMat<sn, 1> cS0(mS0ptr + ib * n, n, 1),
                    cSk(mSkptr + ib * n, n, 1);
#define DEF(x) EigenMat<sn, sn> c##x##T(ptr_offset(m##x##ptr, ib_off), n, n)
            DEF(Uk);
            DEF(Wk);
            DEF(Mk);
            DEF(Mbiask);
            DEF(U0);
            DEF(W0);
            DEF(Bu);
            DEF(Bw);
#undef DEF
            cV0.noalias() = cW0T * cU0T.transpose();
            tmp0 = cMkT - cMbiaskT;
            mprod.init(cV0.transpose(), tmp0).mul_r_to(eqbT, cU0T.transpose());
            {
                // solve Wk
                auto& rhs = tmp0;  // right hand side of the equation
                tmp0 = eqbT.transpose();
                rhs = tmp0 - eqbT;
                rhs -= mprod.init(cV0.transpose(), cBwT.transpose())
                               .mul_r(cV0)
                               .mul_r(cS0.asDiagonal())
                               .get();
                auto& x = rhs;
                for (size_t j = 0; j < n; ++j) {
                    for (size_t i = 0; i < n; ++i) {
                        x(i, j) = clip_div(rhs(i, j), cS0(i) + cS0(j));
                    }
                }
                // update eqbT for future solves
                if (mBu) {
                    eqbT -= mprod.init(x.transpose(), cS0.asDiagonal()).get();
                }
                // solve cWkT
                mprod.init(cV0, x.transpose()).mul_r_to(cWkT, cU0T);
            }
            if (!mBu) {
                continue;
            }
            eqbT.noalias() += cBuT * cS0.asDiagonal();
            cSk = eqbT.diagonal();
            {
                // solve Uk
                auto& cUkTU0 = eqbT;  // Uk.T * U0
                for (size_t j = 0; j < n; ++j) {
                    for (size_t i = 0; i < j; ++i) {
                        fp_t v = clip_div(eqbT(i, j), cS0(i) - cS0(j));
                        cUkTU0(i, j) = v;
                        cUkTU0(j, i) = -cBuT(j, i) - v;
                    }
                    cUkTU0(j, j) = -cBuT(j, j) / 2;
                }
                cUkT.noalias() = cUkTU0 * cU0T;
            }
        }
    };
    const size_t batch = mMk.shape(0), n = mMk.shape(1);
    sanm_assert(n >= 2);
    if (mBu) {
        mUk.set_shape({batch, n, n});
        mSk.set_shape({batch, n});
    }
    mWk.set_shape({batch, n, n});
#define CASE(sn)                  \
    do {                          \
        if (n == sn) {            \
            run.operator()<sn>(); \
            return;               \
        }                         \
    } while (0)
    CASE(2);
    CASE(3);
#undef CASE
    run.operator()<Eigen::Dynamic>();
}

void sanm::svd_w_taylor_fwd_p(TensorND& mPk, TensorND& mWk, const TensorND& mMk,
                              const TensorND& mU0, const TensorND& mS0,
                              const TensorND& mW0, const TensorND& mBm,
                              const TensorND& mBp, const TensorND& mBpw) {
    SANM_SCOPED_PROFILER("svd_w_taylor_fwd_p");
    auto run = [&]<Eigen::Index sn>() {
        const size_t batch = mMk.shape(0), n = mMk.shape(1);
        if (sn != Eigen::Dynamic && static_cast<Eigen::Index>(n) != sn) {
            // hint the compiler that n equals to sn
            __builtin_unreachable();
        }
        fp_t *mPkptr = mPk.woptr(), *mWkptr = mWk.woptr();
#define DEF(x) auto x##ptr = const_cast<fp_t*>(x.ptr())
        DEF(mMk);
        DEF(mU0);
        DEF(mS0);
        DEF(mW0);
        DEF(mBm);
        DEF(mBp);
        DEF(mBpw);
#undef DEF

        // matrix variables:
        // prefix c means current (this batch)
        // suffix t/T means transposed (eigen uses col-major)

        // allocate temporaries
        ScopedAllowMalloc scoped_allow_malloc;
        Eigen::Matrix<fp_t, sn, sn> cV0(n, n), eqbT(n, n);
        Eigen::Matrix<fp_t, sn, 1> cS0inv(n, 1);
        MatrixMultiProduct<Eigen::Matrix<fp_t, sn, sn>> mprod(n, n);
        scoped_allow_malloc.disallow();

        for (size_t ib = 0; ib < batch; ++ib) {
            size_t ib_off = ib * n * n;
            EigenMat<sn, 1> cS0(mS0ptr + ib * n, n, 1);
#define DEF(x) EigenMat<sn, sn> c##x##T(m##x##ptr + ib_off, n, n)
            DEF(Pk);
            DEF(Wk);
            DEF(Mk);
            DEF(U0);
            DEF(W0);
            DEF(Bm);
            DEF(Bp);
            DEF(Bpw);
#undef DEF
            cV0.noalias() = cW0T * cU0T.transpose();
            eqbT = cBmT - cBpT;
            mprod.init(cU0T, eqbT).mul_r_to(eqbT, cU0T.transpose());
            mprod.init(cS0.asDiagonal(), cV0.transpose())
                    .mul_r(cMkT)
                    .mul_r(cU0T.transpose());
            eqbT += mprod.get();
            eqbT += mprod.get().transpose();
            auto& x = eqbT;
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    x(i, j) = clip_div(eqbT(i, j), cS0(i) + cS0(j));
                }
            }
            for (size_t i = 0; i < n; ++i) {
                cS0inv(i) = clip_div(1, cS0(i));
            }
            mprod.init(cU0T.transpose(), x).mul_r_to(cPkT, cU0T);
            eqbT = cMkT - cBpwT;
            eqbT.noalias() -= cW0T * cPkT;
            mprod.init(eqbT, cU0T.transpose())
                    .mul_r(cS0inv.asDiagonal())
                    .mul_r_to(cWkT, cU0T);
        }
    };
    const size_t batch = mMk.shape(0), n = mMk.shape(1);
    sanm_assert(n >= 2);
    mPk.set_shape({batch, n, n});
    mWk.set_shape({batch, n, n});
#define CASE(sn)                  \
    do {                          \
        if (n == sn) {            \
            run.operator()<sn>(); \
            return;               \
        }                         \
    } while (0)
    CASE(2);
    CASE(3);
#undef CASE
    run.operator()<Eigen::Dynamic>();
}
