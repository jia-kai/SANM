/**
 * \file libsanm/tensor_elemwise.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/tensor_impl_helper.h"

using namespace sanm;

namespace {

TensorShape deduce_shape(const TensorShape& s0, const TensorShape& s1) {
    sanm_assert(s0.rank && s1.rank);
    if (s0 == s1) {
        return s0;
    }
    if (s0.is_single_scalar()) {
        return s1;
    }
    if (s1.is_single_scalar()) {
        return s0;
    }
    if (s0.dim[0] == s1.dim[0]) {
        if (s0.is_batched_scalar()) {
            return s1;
        }
        if (s1.is_batched_scalar()) {
            return s0;
        }
    }
    if (s0.rank == s1.rank) {
        TensorShape ret;
        ret.rank = s0.rank;
        bool succ = true;
        for (size_t i = 0; i < s0.rank; ++i) {
            size_t d0 = s0.dim[i], d1 = s1.dim[i];
            if (d0 == d1) {
                ret.dim[i] = d0;
            } else if (d0 == 1 || d1 == 1) {
                ret.dim[i] = std::max(d0, d1);
            } else {
                succ = false;
                break;
            }
        }
        if (succ) {
            return ret;
        }
    }
    throw SANMError{ssprintf("tensor shape mismatch for elemwise op: %s vs %s",
                             s0.str().c_str(), s1.str().c_str())};
}

/*!
 * \brief collapse non-broadcasting dimensions
 * Note that shapes[0] must be the output shape
 * \return rank after collapsing
 */
size_t collapse_non_bcast(std::vector<TensorShape>& shapes) {
    size_t rank = shapes[0].rank;
    for (auto&& i : shapes) {
        if (i.is_batched_scalar()) {
            // broadcast both single scalar and batched scalar
            for (size_t j = i.rank; j < rank; ++j) {
                i.dim[j] = 1;
            }
            i.rank = rank;
        }
        sanm_assert(i.rank == rank);
    }

    auto same = [&shapes](size_t d) {
        size_t s0 = shapes[0].dim[d];
        for (size_t i = 1; i < shapes.size(); ++i) {
            if (shapes[i].dim[d] != s0) {
                return false;
            }
        }
        return true;
    };

    for (size_t r = 0; r < rank;) {
        size_t r1 = r, tot = 1;
        while (r1 < rank && same(r1)) {
            tot *= shapes[0].dim[r1];
            ++r1;
        }
        if (r1 - r > 1) {
            auto dr = r1 - r - 1;
            rank -= dr;
            for (auto&& s : shapes) {
                s.dim[r] = tot;
                for (size_t i = r + 1; i < rank; ++i)
                    s.dim[i] = s.dim[i + dr];
            }
        }
        r = r1 + 1;  // directly skip r1 which has different shapes
    }
    for (auto&& s : shapes) {
        s.rank = rank;
    }
    return rank;
}

//! get strides for each dimension while treating broadcasting as zero stride
TensorShape stride_with_bcast(const TensorShape& shape) {
    TensorShape stride;
    stride.rank = shape.rank;
    size_t mul = 1;
    for (int i = shape.rank - 1; i >= 0; --i) {
        stride.dim[i] = shape.dim[i] == 1 ? 0 : mul;
        mul *= shape.dim[i];
    }
    return stride;
}

struct ConstThinTensor {
    const fp_t* ptr;
    TensorShape shape;

    Eigen::Map<Eigen::Matrix<fp_t, Eigen::Dynamic, 1>> as_vector() {
        return {const_cast<fp_t*>(ptr),
                static_cast<Eigen::Index>(shape.total_nr_elems())};
    }
};

struct ThinTensor {
    fp_t* ptr;
    TensorShape shape;

    Eigen::Map<Eigen::Matrix<fp_t, Eigen::Dynamic, 1>> as_vector() {
        return {ptr, static_cast<Eigen::Index>(shape.total_nr_elems())};
    }
};

template <class OpFunc, bool accum, bool lhs_scalar = false,
          bool rhs_scalar = false>
void compute_elemwise_binary_thin(ThinTensor dst, ConstThinTensor lhs,
                                  ConstThinTensor rhs, OpFunc op_func) {
    if (dst.shape == lhs.shape && dst.shape == rhs.shape) {
        if constexpr (accum) {
            dst.as_vector().array() +=
                    op_func(lhs.as_vector().array(), rhs.as_vector().array());
        } else {
            dst.as_vector().array() =
                    op_func(lhs.as_vector().array(), rhs.as_vector().array());
        }
        return;
    }

    std::vector<TensorShape> shapes{dst.shape, lhs.shape, rhs.shape};
    size_t rank = collapse_non_bcast(shapes);

    TensorShape lhs_stride = stride_with_bcast(shapes[1]),
                rhs_stride = stride_with_bcast(shapes[2]);
    const fp_t *lptr = lhs.ptr, *rptr = rhs.ptr;

    if (rank == 2) {
        size_t size0 = shapes[0][0], size1 = shapes[0][1],
               lstrd0 = lhs_stride[0], lstrd1 = lhs_stride[1],
               rstrd0 = rhs_stride[0], rstrd1 = rhs_stride[1];
        for (size_t i = 0; i < size0; ++i) {
            for (size_t j = 0; j < size1; ++j) {
                fp_t v = op_func(
                        lhs_scalar ? lptr[0] : lptr[i * lstrd0 + j * lstrd1],
                        rhs_scalar ? rptr[0] : rptr[i * rstrd0 + j * rstrd1]);
                fp_t& d = dst.ptr[i * size1 + j];
                if constexpr (accum) {
                    d += v;
                } else {
                    d = v;
                }
            }
        }
        return;
    }
    if (rank == 3) {
        size_t size0 = shapes[0][0], size1 = shapes[0][1], size2 = shapes[0][2],
               lstrd0 = lhs_stride[0], lstrd1 = lhs_stride[1],
               lstrd2 = lhs_stride[2], rstrd0 = rhs_stride[0],
               rstrd1 = rhs_stride[1], rstrd2 = rhs_stride[2];
        for (size_t i = 0; i < size0; ++i) {
            for (size_t j = 0; j < size1; ++j) {
                for (size_t k = 0; k < size2; ++k) {
                    fp_t v = op_func(lhs_scalar ? lptr[0]
                                                : lptr[i * lstrd0 + j * lstrd1 +
                                                       k * lstrd2],
                                     rhs_scalar ? rptr[0]
                                                : rptr[i * rstrd0 + j * rstrd1 +
                                                       k * rstrd2]);
                    fp_t& d = dst.ptr[(i * size1 + j) * size2 + k];
                    if constexpr (accum) {
                        d += v;
                    } else {
                        d = v;
                    }
                }
            }
        }
        return;
    }
    throw SANMError{ssprintf("unhandled rank %zu: shapes: dst=%s l=%s r=%s",
                             rank, dst.shape.str().c_str(),
                             lhs.shape.str().c_str(), rhs.shape.str().c_str())};
}

template <class OpFunc, bool accum>
void compute_elemwise_binary(TensorND& dst, const TensorND& lhs,
                             const TensorND& rhs, OpFunc op_func = {}) {
    sanm_assert(!lhs.same_storage(rhs) || lhs.shape() == rhs.shape(),
                "shape must match if tensors share storage");
    auto out_shape = deduce_shape(lhs.shape(), rhs.shape());
    bool dst_rw_mode;
    if (bool sl = (&dst == &lhs), sr = (&dst == &rhs); sl || sr) {
        sanm_assert((!sl || out_shape == lhs.shape()) &&
                            (!sr || out_shape == rhs.shape()),
                    "inplace elemwise shape mismatch: %s vs %s",
                    lhs.shape().str().c_str(), rhs.shape().str().c_str());
        dst_rw_mode = true;
    } else {
        if (accum) {
            if (dst.shape() != out_shape) {
                // dst can be larger than the elements in the accum mode
                TensorShape compat = deduce_shape(dst.shape(), out_shape);
                sanm_assert(dst.shape() == compat,
                            "accum dst shape mismatch: %s vs %s",
                            dst.shape().str().c_str(), out_shape.str().c_str());
                out_shape = compat;
            }
            dst_rw_mode = true;
        } else {
            dst.set_shape(out_shape);
            dst_rw_mode = (&dst == &lhs || &dst == &rhs);
        }
    }

    {
        bool lz = lhs.is_zero(), rz = rhs.is_zero();
        if constexpr (OpFunc::OP == '+' || OpFunc::OP == '-') {
            // disable shortcut if shape does not match
            if (lz && dst.shape() != rhs.shape()) {
                lz = false;
            }
            if (rz && dst.shape() != lhs.shape()) {
                rz = false;
            }
        }
        if (lz || rz) {
            if constexpr (OpFunc::OP == '+') {
                const TensorND& val = lz ? rhs : lhs;
                if (accum) {
                    dst += val;
                } else {
                    dst = val;
                }
                return;
            }
            if constexpr (OpFunc::OP == '-') {
                TensorND val = rz ? lhs : -rhs;
                if (accum) {
                    dst += val;
                } else {
                    dst = std::move(val);
                }
                return;
            }
            if constexpr (OpFunc::OP == '*') {
                if (!accum) {
                    dst.fill_with_inplace(0);
                }
                return;
            }
            if constexpr (OpFunc::OP == '/') {
                sanm_assert(!rz, "division by zero");
                if (!accum) {
                    dst.fill_with_inplace(0);
                }
                return;
            }
            throw SANMError{"impossible"};
        }
    }

    fp_t* out_ptr = dst_rw_mode ? dst.rwptr() : dst.woptr();
    compute_elemwise_binary_thin<OpFunc, accum>(
            {out_ptr, dst.shape()}, {lhs.ptr(), lhs.shape()},
            {rhs.ptr(), rhs.shape()}, op_func);
}

//! trait for elemwise binary ops
template <char op>
struct ElemwiseBinOpFunc;

}  // anonymous namespace

#define DEF_TRAIT(ch, op)                                                 \
    namespace {                                                           \
    template <>                                                           \
    struct ElemwiseBinOpFunc<ch> {                                        \
        static constexpr char OP = ch;                                    \
        template <typename A, typename B>                                 \
        auto operator()(A&& lhs, B&& rhs) const {                         \
            return lhs op rhs;                                            \
        }                                                                 \
    };                                                                    \
    }                                                                     \
    template <>                                                           \
    TensorND& TensorND::as_elem<ch>(const TensorND& lhs,                  \
                                    const TensorND& rhs) {                \
        compute_elemwise_binary<ElemwiseBinOpFunc<ch>, false>(*this, lhs, \
                                                              rhs);       \
        return *this;                                                     \
    }                                                                     \
    TensorND TensorND::operator op(const TensorND& rhs) const {           \
        TensorND ret;                                                     \
        ret.as_elem<ch>(*this, rhs);                                      \
        return ret;                                                       \
    }                                                                     \
    TensorND& TensorND::operator op##=(const TensorND& rhs) {             \
        this->as_elem<ch>(*this, rhs);                                    \
        return *this;                                                     \
    }

DEF_TRAIT('+', +);
DEF_TRAIT('-', -);
DEF_TRAIT('*', *);
DEF_TRAIT('/', /);
#undef DEF_TRAIT

struct ElemwiseMulWithScaleOpFunc {
    fp_t scale;
    static constexpr char OP = '*';

    template <typename A, typename B>
    auto operator()(A&& lhs, B&& rhs) const {
        return lhs * rhs * scale;
    }
};

TensorND& TensorND::accum_mul(const TensorND& lhs, const TensorND& rhs,
                              fp_t scale) {
    if (empty()) {
        *this = lhs * rhs * scale;
    } else {
        if (scale == 1.) {
            compute_elemwise_binary<ElemwiseBinOpFunc<'*'>, true>(*this, lhs,
                                                                  rhs);
        } else {
            using OpFunc = ElemwiseMulWithScaleOpFunc;
            OpFunc op_func;
            op_func.scale = scale;
            compute_elemwise_binary<OpFunc, true>(*this, lhs, rhs, op_func);
        }
    }
    return *this;
}

TensorND& TensorND::as_fma(const TensorND& x, const TensorND& y,
                           const TensorND& b) {
    sanm_assert(x.shape() == y.shape() && x.shape() == b.shape(),
                "shape mismatch in FMA: %s %s %s", x.shape().str().c_str(),
                y.shape().str().c_str(), b.shape().str().c_str());
    if (x.is_zero() || y.is_zero()) {
        *this = b;
        return *this;
    }
    if (x.is_one()) {
        return as_elem<'+'>(y, b);
    }
    if (y.is_one()) {
        return as_elem<'+'>(x, b);
    }
    set_shape(x.shape());
    as_vector_w(*this) = as_vector_r(x).array() * as_vector_r(y).array() +
                         as_vector_r(b).array();
    return *this;
}

TensorND& TensorND::accum_mul(const TensorND& lhs, fp_t rhs) {
    if (empty()) {
        return *this = lhs * rhs;
    }
    if (is_zero() && m_shape == lhs.shape()) {
        return *this = lhs * rhs;
    }
    auto tot_shape = deduce_shape(m_shape, lhs.shape());
    sanm_assert(tot_shape == m_shape,
                "can not accum tensor of shape %s into %s",
                lhs.shape().str().c_str(), m_shape.str().c_str());
    if (lhs.is_zero() || rhs == 0. || rhs == -0.) {
        return *this;
    }
    if (tot_shape == lhs.shape()) {
        as_vector_w(*this).array() += as_vector_r(lhs).array() * rhs;
        return *this;
    }
    ConstThinTensor rhs_t;
    rhs_t.ptr = &rhs;
    rhs_t.shape.rank = tot_shape.rank;
    for (size_t i = 0; i < rhs_t.shape.rank; ++i) {
        rhs_t.shape.dim[i] = 1;
    }
    using OpFunc = ElemwiseBinOpFunc<'*'>;
    compute_elemwise_binary_thin<OpFunc, true, false, true>(
            {rwptr(), m_shape}, {lhs.ptr(), lhs.shape()}, rhs_t, OpFunc{});
    return *this;
}
