/**
 * \file libsanm/tensor_linalg.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/tensor_impl_helper.h"

#include <Eigen/Dense>

using namespace sanm;

namespace {
/*!
 * \brief compute cofactor of a single matrix
 * \param s_prod_rev a workspace of size dim+1, and the last value must be
 *      initialized to one
 */
template <int mat_dim>
void compute_cofactor(EigenMat<mat_dim, mat_dim>& mdst,
                      EigenMat<mat_dim, mat_dim>& msrc, fp_t* s_prod_rev) {
    Eigen::Index dim = msrc.rows();

    Eigen::JacobiSVD<Eigen::Matrix<fp_t, mat_dim, mat_dim>> svd{
            msrc, Eigen::ComputeFullU | Eigen::ComputeFullV};

    if (svd.rank() + 2 <= dim) {
        mdst.setZero();
        return;
    }

    // det(S) * S.inv()
    Eigen::Matrix<fp_t, mat_dim, 1> sinvd = svd.singularValues();
    if constexpr (mat_dim == 2) {
        fp_t a = sinvd(0), b = sinvd(1);
        sinvd(0) = b;
        sinvd(1) = a;
    } else if constexpr (mat_dim == 3) {
        fp_t a = sinvd(0), b = sinvd(1), c = sinvd(2);
        sinvd(0) = b * c;
        sinvd(1) = a * c;
        sinvd(2) = a * b;
    } else {
        for (int i = dim - 1; i >= 0; --i) {
            s_prod_rev[i] = s_prod_rev[i + 1] * sinvd[i];
        }
        fp_t prod = 1;
        for (Eigen::Index i = 0; i < dim; ++i) {
            fp_t orig = sinvd(i);
            sinvd(i) = prod * s_prod_rev[i + 1];
            prod *= orig;
        }
    }

    fp_t sign = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    if (sign < 0) {
        sinvd = -sinvd;
    }
    mdst = svd.matrixU() * sinvd.asDiagonal() * svd.matrixV().transpose();
}

template <typename T>
constexpr T get_from_pair(int d, T x, T y) {
    return d == 0 ? x : y;
}

CBLAS_TRANSPOSE cblas_trans(bool t) {
    return t ? CblasTrans : CblasNoTrans;
}
}  // anonymous namespace

TensorND& TensorND::as_mm(const TensorND& lhs, const TensorND& rhs, bool accum,
                          bool trans_lhs, bool trans_rhs) {
    SANM_SCOPED_PROFILER("mm");
    sanm_assert(this != &lhs && this != &rhs);
    sanm_assert(lhs.rank() == 2 && rhs.rank() == 2);
    TensorShape dst_shape{
            get_from_pair(trans_lhs, lhs.shape(0), lhs.shape(1)),
            get_from_pair(trans_rhs ^ 1, rhs.shape(0), rhs.shape(1))};

    if (accum) {
        sanm_assert(shape() == dst_shape, "mm accum: expect shape %s, got %s",
                    dst_shape.str().c_str(), shape().str().c_str());

        if (lhs.is_zero() || rhs.is_zero()) {
            return *this;
        }
    } else {
        if (lhs.is_zero() || rhs.is_zero()) {
            // do not use set_shape to avoid unnecessary memory allocation
            m_shape = dst_shape;
            return fill_with_inplace(0);
        }
        set_shape(dst_shape);
    }

    static_assert(std::is_same_v<fp_t, double>, "unhandled fp_t");
    if constexpr (std::is_same_v<fp_t, double>) {
        cblas_dgemm(CblasRowMajor, cblas_trans(trans_lhs),
                    cblas_trans(trans_rhs), dst_shape[0], dst_shape[1],
                    get_from_pair(trans_lhs ^ 1, lhs.shape(0), lhs.shape(1)), 1,
                    lhs.ptr(), lhs.shape(1), rhs.ptr(), rhs.shape(1),
                    accum ? 1 : 0, rwptr(), dst_shape[1]);
    }
    return *this;
}

TensorND& TensorND::as_batched_mm(const TensorND& lhs, const TensorND& rhs,
                                  bool accum, bool trans_lhs, bool trans_rhs) {
    SANM_SCOPED_PROFILER("batched_mm");
    sanm_assert(this != &lhs && this != &rhs);
    sanm_assert(
            lhs.rank() == 3 && rhs.rank() == 3 && lhs.shape(0) == rhs.shape(0),
            "batched mm shape mismatch %s vs %s", lhs.shape().str().c_str(),
            rhs.shape().str().c_str());
    const size_t ls0 = lhs.shape(1), ls1 = lhs.shape(2), rs0 = rhs.shape(1),
                 rs1 = rhs.shape(2), batch = lhs.shape(0);
    sanm_assert(get_from_pair(trans_lhs ^ 1, ls0, ls1) ==
                        get_from_pair(trans_rhs, rs0, rs1),
                "matmul shape mismatch: %s vs %s, trans=%d,%d",
                lhs.shape().str().c_str(), rhs.shape().str().c_str(), trans_lhs,
                trans_rhs);

    TensorShape dst_shape{batch, get_from_pair(trans_lhs, ls0, ls1),
                          get_from_pair(trans_rhs ^ 1, rs0, rs1)};

    if (accum) {
        sanm_assert(shape() == dst_shape,
                    "batched_mm accum: expect shape %s, got %s",
                    dst_shape.str().c_str(), shape().str().c_str());

        if (lhs.is_zero() || rhs.is_zero()) {
            return *this;
        }
    } else {
        if (lhs.is_zero() || rhs.is_zero()) {
            // do not use set_shape to avoid unnecessary memory allocation
            m_shape = dst_shape;
            return fill_with_inplace(0);
        }
        set_shape(dst_shape);
    }

    // As of 2020, MKL batched dgemm is still slow for small matrices ...
    // So we roll out our simple accelerated impl using eigen

    auto pa = const_cast<fp_t*>(lhs.ptr()), pb = const_cast<fp_t*>(rhs.ptr()),
         pc = rwptr();
    auto run_static_shape = [pa, pb, pc, batch, accum, trans_lhs,
                             trans_rhs]<int ls0, int ls1, int rs0, int rs1>() {

    // eigen uses col-major; so we compute on the transformed matrices
#define FOREACH_8(cb)                                                       \
    cb(0, 0, 0) cb(0, 0, 1) cb(0, 1, 0) cb(0, 1, 1) cb(1, 0, 0) cb(1, 0, 1) \
            cb(1, 1, 0) cb(1, 1, 1)
#define ACC0 =
#define ACC1 +=
#define TR0(x) x
#define TR1(x) x.transpose()
#define CB(acc, tra, trb)                                               \
    if constexpr (get_from_pair(tra ^ 1, ls0, ls1) ==                   \
                  get_from_pair(trb, rs0, rs1)) {                       \
        constexpr int cs0 = get_from_pair(tra, ls0, ls1),               \
                      cs1 = get_from_pair(trb ^ 1, rs0, rs1);           \
        if (accum == acc && trans_lhs == tra && trans_rhs == trb) {     \
            for (size_t ib = 0; ib < batch; ++ib) {                     \
                EigenMat<ls1, ls0> ta{pa + ib * (ls0 * ls1), ls1, ls0}; \
                EigenMat<rs1, rs0> tb{pb + ib * (rs0 * rs1), rs1, rs0}; \
                EigenMat<cs1, cs0> tc{pc + ib * (cs0 * cs1), cs1, cs0}; \
                tc.noalias() ACC##acc TR##trb(tb) * TR##tra(ta);        \
            }                                                           \
            return;                                                     \
        }                                                               \
    }
        FOREACH_8(CB)
        sanm_assert(0, "impossible");
#undef CB
#undef TR1
#undef TR0
#undef ACC1
#undef ACC0
#undef FOREACH_8
    };

#define CASE(ls0_, ls1_, rs0_, rs1_)                                    \
    do {                                                                \
        if (ls0_ == ls0 && ls1_ == ls1 && rs0_ == rs0 && rs1_ == rs1) { \
            run_static_shape.operator()<ls0_, ls1_, rs0_, rs1_>();      \
            return *this;                                               \
        }                                                               \
    } while (0)

    CASE(1, 1, 1, 1);
    CASE(2, 2, 2, 2);
    CASE(3, 3, 3, 3);
    CASE(4, 4, 4, 4);

    // special handling of shapes can be added here
#undef CASE

    static_assert(std::is_same_v<fp_t, double>, "unhandled fp_t");
    if constexpr (std::is_same_v<fp_t, double>) {
        cblas_dgemm_batch_strided(
                CblasRowMajor, cblas_trans(trans_lhs), cblas_trans(trans_rhs),
                dst_shape[1], dst_shape[2],
                get_from_pair(trans_lhs ^ 1, ls0, ls1), 1, lhs.ptr(), ls1,
                ls0 * ls1, rhs.ptr(), rs1, rs0 * rs1, accum ? 1 : 0, rwptr(),
                dst_shape[2], dst_shape[1] * dst_shape[2], batch);
    }
    return *this;
}

TensorND& TensorND::as_batched_transpose(const TensorND& src) {
    SANM_SCOPED_PROFILER("batched_transpose");
    sanm_assert(this != &src);
    sanm_assert(src.rank() == 3);
    {
        TensorShape oshp = src.shape();
        std::swap(oshp.dim[1], oshp.dim[2]);
        set_shape(oshp);
    }
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    auto work = [this, &src]<int mat_dim0, int mat_dim1>() {
        auto sptr = const_cast<fp_t*>(src.ptr()), dptr = this->woptr();
        const size_t batch = src.shape(0);
        const Eigen::Index dim0 = src.shape(1), dim1 = src.shape(2);
        for (size_t i = 0; i < batch; ++i) {
            EigenMat<mat_dim0, mat_dim1> mdst{dptr + i * dim0 * dim1, dim0,
                                              dim1};
            EigenMat<mat_dim1, mat_dim0> msrc{sptr + i * dim0 * dim1, dim1,
                                              dim0};
            mdst = msrc.transpose();
        }
    };

    auto work_dispatch = [&work, &src]<int mat_dim0>() {
        switch (src.shape(2)) {
#define ON(x)                           \
    case x:                             \
        work.operator()<mat_dim0, x>(); \
        break
            ON(1);
            ON(2);
            ON(3);
            ON(4);
#undef ON
            default:
                work.operator()<mat_dim0, Eigen::Dynamic>();
                break;
        }
    };

    switch (src.shape(1)) {
#define ON(x)                          \
    case x:                            \
        work_dispatch.operator()<x>(); \
        break
        ON(1);
        ON(2);
        ON(3);
        ON(4);
#undef ON
        default:
            work_dispatch.operator()<Eigen::Dynamic>();
            break;
    }
    return *this;
}

TensorND& TensorND::as_transpose(const TensorND& src) {
    sanm_assert(src.rank() == 2);
    size_t m = src.shape(0), n = src.shape(1);
    set_shape({n, m});
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    auto ei = [](size_t x) -> Eigen::Index { return x; };
    EigenMatDyn mdst{woptr(), ei(m), ei(n)},
            msrc{const_cast<fp_t*>(src.ptr()), ei(n), ei(m)};
    mdst = msrc.transpose();
    return *this;
}

TensorND& TensorND::as_batched_matinv(const TensorND& src) {
    SANM_SCOPED_PROFILER("batched_matinv");
    sanm_assert(this != &src);
    sanm_assert(src.rank() == 3 && src.shape(1) == src.shape(2));
    sanm_assert(!src.is_zero());
    set_shape(src.shape());
    auto work = [this, &src]<int mat_dim>() {
        ScopedAllowMalloc scoped_allow_malloc;
        auto sptr = const_cast<fp_t*>(src.ptr()), dptr = this->woptr();
        const size_t batch = src.shape(0);
        const Eigen::Index dim = src.shape(1);
        for (size_t i = 0; i < batch; ++i) {
            EigenMat<mat_dim, mat_dim> mdst{dptr + i * dim * dim, dim, dim},
                    msrc{sptr + i * dim * dim, dim, dim};
            mdst = msrc.inverse();
        }
    };
    switch (src.shape(1)) {
#define ON(x)                 \
    case x:                   \
        work.operator()<x>(); \
        break
        ON(1);
        ON(2);
        ON(3);
        ON(4);
#undef ON
        default:
            work.operator()<Eigen::Dynamic>();
            break;
    }
    return *this;
}

TensorND& TensorND::as_batched_determinant(const TensorND& src) {
    SANM_SCOPED_PROFILER("batched_det");
    ScopedAllowMalloc allow_mem_alloc;

    sanm_assert(this != &src);
    sanm_assert(src.rank() == 3 && src.shape(1) == src.shape(2));
    set_shape({src.shape(0), 1});
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    auto work = [this, &src]<int mat_dim>() {
        auto sptr = const_cast<fp_t*>(src.ptr()), dptr = this->woptr();
        const size_t batch = src.shape(0);
        const Eigen::Index dim = src.shape(1);
        for (size_t i = 0; i < batch; ++i) {
            EigenMat<mat_dim, mat_dim> msrc{sptr + i * dim * dim, dim, dim};
            dptr[i] = msrc.determinant();
        }
    };
    switch (src.shape(1)) {
#define ON(x)                 \
    case x:                   \
        work.operator()<x>(); \
        break
        ON(1);
        ON(2);
        ON(3);
        ON(4);
#undef ON
        default:
            work.operator()<Eigen::Dynamic>();
            break;
    }
    return *this;
}

TensorND& TensorND::as_batched_cofactor(const TensorND& src) {
    SANM_SCOPED_PROFILER("batched_cofactor");
    ScopedAllowMalloc allow_mem_alloc;

    sanm_assert(this != &src);
    sanm_assert(src.rank() == 3 && src.shape(1) == src.shape(2) &&
                src.shape(1) >= 2);
    set_shape(src.shape());
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    auto work = [this, &src]<int mat_dim>() {
        auto sptr = const_cast<fp_t*>(src.ptr()), dptr = this->woptr();
        const size_t batch = src.shape(0);
        const Eigen::Index dim = src.shape(1);
        std::unique_ptr<fp_t[]> s_prod_rev{new fp_t[dim + 1]};
        s_prod_rev[dim] = 1;
        for (size_t i = 0; i < batch; ++i) {
            EigenMat<mat_dim, mat_dim> mdst{dptr + i * dim * dim, dim, dim},
                    msrc{sptr + i * dim * dim, dim, dim};
            compute_cofactor(mdst, msrc, s_prod_rev.get());
        }
    };
    switch (src.shape(1)) {
#define ON(x)                 \
    case x:                   \
        work.operator()<x>(); \
        break
        ON(2);
        ON(3);
        ON(4);
#undef ON
        default:
            work.operator()<Eigen::Dynamic>();
            break;
    }
    return *this;
}

TensorND& TensorND::as_batched_mm_vecitem_left(const TensorND& lhs,
                                               const TensorND& rhs,
                                               bool accum) {
    SANM_SCOPED_PROFILER("batched_mm_vecitem");
    sanm_assert(this != &lhs && this != &rhs);
    sanm_assert(lhs.rank() == 3 && rhs.rank() == 3);
    size_t BATCH = lhs.shape(0), M = lhs.shape(1), P = lhs.shape(2),
           K = rhs.shape(1), N = rhs.shape(2);
    sanm_assert(M % K == 0 && BATCH == rhs.shape(0));
    M /= K;

    TensorShape expect_shape{BATCH, M * N, P};
    if (accum) {
        sanm_assert(m_shape == expect_shape);
    } else {
        set_shape(expect_shape);
    }

    if (lhs.is_zero() || rhs.is_zero()) {
        return fill_with_inplace(0);
    }

    auto ei = [](size_t x) -> Eigen::Index { return x; };

    auto lptr = const_cast<fp_t*>(lhs.ptr()),
         rptr = const_cast<fp_t*>(rhs.ptr()), optr = this->woptr();
    for (size_t b = 0; b < BATCH; ++b) {
        EigenMatDyn mat_rhs{rptr + b * K * N, ei(N), ei(K)};
        for (size_t m = 0; m < M; ++m) {
            EigenMatDyn mat_lhs{lptr + (b * M + m) * K * P, ei(P), ei(K)},
                    mat_dst{optr + (b * M + m) * N * P, ei(P), ei(N)};
            if (accum) {
                mat_dst.noalias() += mat_lhs * mat_rhs.transpose();
            } else {
                mat_dst.noalias() = mat_lhs * mat_rhs.transpose();
            }
        }
    }

    return *this;
}
