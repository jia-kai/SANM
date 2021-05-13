/**
 * \file libsanm/tensor_impl_helper.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// this file contains helpers to implement tenor functions

#pragma once

#include "libsanm/tensor.h"

#define EIGEN_RUNTIME_NO_MALLOC 1
#include <Eigen/Core>

#include <atomic>

namespace sanm {
using EigenVec = Eigen::Map<Eigen::Matrix<fp_t, Eigen::Dynamic, 1>>;
using EigenCVec = Eigen::Map<const Eigen::Matrix<fp_t, Eigen::Dynamic, 1>>;

template <int dim0, int dim1>
using EigenMat = Eigen::Map<Eigen::Matrix<fp_t, dim0, dim1>>;
using EigenMatDyn = EigenMat<Eigen::Dynamic, Eigen::Dynamic>;

//! reset the internal buffer of an EigenVec
static inline EigenVec& reset(EigenVec& v, const fp_t* ptr, size_t len) {
    v.~EigenVec();
    new (&v)
            EigenVec{const_cast<fp_t*>(ptr), static_cast<Eigen::Index>(len), 1};
    return v;
}

static inline EigenVec as_vector_r(const TensorND& t) {
    sanm_assert(t.rank());
    return {const_cast<fp_t*>(t.ptr()),
            static_cast<Eigen::Index>(t.shape().total_nr_elems())};
}

static inline EigenVec as_vector_w(TensorND& t) {
    sanm_assert(t.rank());
    return {t.rwptr(), static_cast<Eigen::Index>(t.shape().total_nr_elems())};
}

//! allow eigen to malloc memory in the lifespan of this class
class ScopedAllowMalloc {
    static std::atomic_size_t sm_stack_depth;
    bool m_valid = true;

public:
    ScopedAllowMalloc() {
        if (sm_stack_depth.fetch_add(1) == 0) {
            Eigen::internal::set_is_malloc_allowed(true);
        }
    }
    void disallow() {
        if (m_valid) {
            m_valid = false;
            if (sm_stack_depth.fetch_sub(1) == 1) {
                Eigen::internal::set_is_malloc_allowed(false);
            }
        }
    }
    ~ScopedAllowMalloc() { disallow(); }
};

//! compute the product of many matrices by preallocating and reusing the
//! temporary storage
template <class MatrixType>
class MatrixMultiProduct : public NonCopyable {
    MatrixType m_v0, m_v1, *m_v[2];
    int m_cur = 0;

public:
    MatrixMultiProduct(int rows, int cols)
            : m_v0(rows, cols), m_v1(rows, cols) {
        m_v[0] = &m_v0;
        m_v[1] = &m_v1;
    }

    template <class L, class R>
    MatrixMultiProduct& init(const L& lhs, const R& rhs) {
        m_v0.noalias() = lhs * rhs;
        m_cur = 0;
        return *this;
    }

    template <class L>
    MatrixMultiProduct& mul_l(const L& lhs) {
        m_v[m_cur ^ 1]->noalias() = lhs * (*m_v[m_cur]);
        m_cur ^= 1;
        return *this;
    }

    template <class R>
    MatrixMultiProduct& mul_r(const R& rhs) {
        m_v[m_cur ^ 1]->noalias() = (*m_v[m_cur]) * rhs;
        m_cur ^= 1;
        return *this;
    }

    template <class M, class R>
    const MatrixMultiProduct& mul_r_to(M& dst, const R& rhs) const {
        dst.noalias() = (*m_v[m_cur]) * rhs;
        return *this;
    }

    const MatrixType& get() const { return *m_v[m_cur]; }
};
}  // namespace sanm
