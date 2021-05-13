/**
 * \file libsanm/tensor.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/typedefs.h"
#include "libsanm/utils.h"

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

namespace sanm {
//! shape of a tensor
struct TensorShape {
    static constexpr size_t MAX_RANK = 3;

    /*!
     * Note that a rank of 0 means an uninitialized tensor. Scalars are
     * represented by a shape [1] tensor with rank=1. Since order refers to the
     * order of expansion polynomial in the context of ANM, we use the term
     * "rank" to refer to the dimension of a tensor.
     */
    leastsize_t rank = 0;
    leastsize_t dim[MAX_RANK];

    TensorShape() = default;
    TensorShape(std::initializer_list<size_t> sizes);

    //! flatten all non-batch dimensions (the result shape has rank 2, first dim
    //! unchanged)
    TensorShape flatten_batched() const;

    //! flatten to a vector (the result shape has rank 1)
    TensorShape flatten_as_vec() const;

    //! return a tensor shape with another batch size
    TensorShape with_batch_replaced(size_t batch) const;

    //! total number of elements
    size_t total_nr_elems() const;

    //! total number of elements per batch (i.e., prod(dim[1:]))
    size_t total_nr_elems_per_batch() const;

    bool operator==(const TensorShape& rhs) const;

    size_t operator[](size_t i) const {
        sanm_assert(i < rank);
        return dim[i];
    }

    //! whether this shape represents batched scalars (including true scalar)
    bool is_batched_scalar() const {
        return rank == 1 || (rank == 2 && dim[1] == 1);
    }

    //! whether this shape represents a scalar (i.e., shape[1])
    bool is_single_scalar() const { return rank == 1 && dim[0] == 1; }

    //! add a new axis; all existing axes starting at \p axis will be moved one
    //! axis forward
    TensorShape add_axis(size_t axis) const;

    //! return a new shape with the given axis removed
    TensorShape remove_axis(size_t axis) const;

    std::string str() const;
};

class TensorStorage : public NonCopyable {
    fp_t* m_data = nullptr;
    size_t m_size = 0;

public:
    ~TensorStorage();

    /*!
     * \brief ensure that the storage holds \p size elements of type fp_t
     *
     * There is a pre-allocation strategy.
     *
     * \return whether size is changed
     */
    void ensure_size(size_t size);

    fp_t* ptr() const { return m_data; }

    size_t size() const { return m_size; }

    enum class Constant {
        ZERO,
        ONE,
    };

    /*!
     * \brief a special storage type to represent all-constant values that can
     *      be shared by multiple TensorND instances;
     * \param size required storage size in the number of elements
     *
     * This function is thread safe. Do not change the returned storage object.
     */
    template <Constant val>
    static const std::shared_ptr<const TensorStorage>& constant(
            size_t size = 0);
};

/*!
 * \brief a simple high-dimensional tensor container.
 *
 * The dimension and rank are dynamic. Different tensors can share the same
 * storage while have different shapes. Copy-on-write is used: tensor storage
 * would be copied before writing if multiple tensors share the same storage.
 *
 * The tensor is stored in row major order (i.e., stride for last dim is 1).
 */
class TensorND {
    std::shared_ptr<TensorStorage> m_storage;
    TensorShape m_shape;

    //! ensure that no other tensor shares the storage of this one by clearing
    //! the storage if it is shared
    TensorND& ensure_unshared_nokeep();

    //! similar to ensure_unshared_nokeep(), but the data is copied to a new
    // storage if it is shared
    TensorND& ensure_unshared_keep();

public:
    TensorND() = default;

    TensorND(const TensorShape& shape) { set_shape(shape); }

    //! reset the tensor to empty shape without deallocating the storage
    void clear() {
        m_shape.rank = 0;
        ensure_unshared_nokeep();
    }

    //! set the shape of this tensor. Original content may be discarded
    TensorND& set_shape(const TensorShape& shape);

    //! return whether these two tensors share the same storage
    bool same_storage(const TensorND& other) const {
        return m_storage.get() == other.m_storage.get();
    }

    //! whether the content of this tensor is all zero
    bool is_zero() const {
        return m_storage.get() ==
               TensorStorage::constant<TensorStorage::Constant::ZERO>().get();
    }

    //! whether the content of this tensor is all one
    bool is_one() const {
        return m_storage.get() ==
               TensorStorage::constant<TensorStorage::Constant::ONE>().get();
    }

    bool empty() const { return m_shape.rank == 0; }

    const TensorShape& shape() const { return m_shape; }
    size_t shape(size_t i) const { return m_shape[i]; }
    size_t shape(size_t i, size_t default_) const {
        return i < m_shape.rank ? m_shape.dim[i] : default_;
    }
    size_t rank() const { return m_shape.rank; }

    //! get a pointer for read-write
    fp_t* rwptr() {
        ensure_unshared_keep();
        sanm_assert(m_storage);
        return m_storage->ptr();
    }

    //! get a pointer for write-only (existing data might be cleared)
    fp_t* woptr() {
        ensure_unshared_nokeep();
        sanm_assert(m_storage);
        return m_storage->ptr();
    }

    //! get a pointer for read
    const fp_t* ptr() const {
        sanm_assert(m_storage);
        return m_storage->ptr();
    }

    /*!
     * \brief Share the storage with \p src, with a new shape
     *
     * The orignal shape and the new shape must have the same number of
     * elements.
     */
    TensorND& as_reshaped(const TensorND& src, const TensorShape& shape);

    TensorND reshape(const TensorShape& shape) const {
        TensorND ret;
        ret.as_reshaped(*this, shape);
        return ret;
    }

    TensorND& reshape_inplace(const TensorShape& shape) {
        return as_reshaped(*this, shape);
    }

    /*!
     * \brief replace a sub batch in this tensor with another batch of \p src,
     *      by computing this_sub = this_sub * alpha + src_sub * beta
     * \param dbatch beginning of dest batch to be overwritten
     * \param sbatch beginning of source batch to be read
     * \param size number of batches to copy
     *
     * Note: if \p alpha is zero and \beta is one, then the sub batch will be
     * replaced (NaN values ignored).
     *
     * When \p dbatch is 0 and this tensor is empty, the size of this tensor
     * will be grown if \p size is too large. Otherwise shape of this tensor
     * must match the required shape.
     */
    TensorND& incr_sub_batch(const TensorND& src, size_t dbatch, size_t sbatch,
                             size_t size, fp_t alpha, fp_t beta);

    //! incr_sub_batch(), with alpha=0 and beta=1
    TensorND& copy_from_sub_batch(const TensorND& src, size_t dbatch,
                                  size_t sbatch, size_t size) {
        return incr_sub_batch(src, dbatch, sbatch, size, 0, 1);
    }

    //! take a sub tensor at the beginning that shares the same storage
    TensorND take_sub(const TensorShape& shape) const;

    //! see TensorShape::flatten_batched()
    TensorND flatten_batched() const {
        return reshape(m_shape.flatten_batched());
    }

    //! see TensorShape::flatten_as_vec()
    TensorND flatten_as_vec() const {
        return reshape(m_shape.flatten_as_vec());
    }

    //! return a new tensor of the same shape and filled with a given value
    TensorND fill_with(fp_t value) const {
        return make_same_shape().fill_with_inplace(value);
    }

    //! fill this tensor with a given scalar value
    TensorND& fill_with_inplace(fp_t value);

    //! make a new tensor with the same shape and undefined content
    TensorND make_same_shape() const { return TensorND{}.set_shape(shape()); }

    //! element-wise multiplication with scalar
    TensorND operator*(fp_t scale) const;
    TensorND& operator*=(fp_t scale);

    TensorND operator/(fp_t x) const { return operator*(1 / x); }
    TensorND& operator/=(fp_t x) { return operator*=(1 / x); }

    //! element-wise addition
    TensorND operator+(const TensorND& rhs) const;
    //! element-wise subtraction
    TensorND operator-(const TensorND& rhs) const;
    //! element-wise multiplication
    TensorND operator*(const TensorND& rhs) const;
    //! element-wise division
    TensorND operator/(const TensorND& rhs) const;

    //! inplace element-wise addition
    TensorND& operator+=(const TensorND& rhs);
    //! inplace element-wise subtraction
    TensorND& operator-=(const TensorND& rhs);
    //! inplace element-wise multiplication
    TensorND& operator*=(const TensorND& rhs);
    //! inplace element-wise division
    TensorND& operator/=(const TensorND& rhs);

    /*!
     * \brief set this as the result of applying the given elemetwise arithmetic
     *      operator on the two operand tensors
     * \tparam op one of '+', '-', '*', '/'
     */
    template <char op>
    TensorND& as_elem(const TensorND& lhs, const TensorND& rhs);

    //! negation operator
    TensorND operator-() const {
        TensorND ret;
        ret.as_neg(*this);
        return ret;
    }

    //! set this as x*y+b; all inputs must have the same shape
    TensorND& as_fma(const TensorND& x, const TensorND& y, const TensorND& b);

    //! elemwise this += lhs * rhs * scale; this must be either empty or the
    //! same shape of the expected result
    TensorND& accum_mul(const TensorND& lhs, const TensorND& rhs,
                        fp_t scale = 1);

    //! elemwise this += lhs * rhs; this must be either empty, the same shape,
    //! or a larger consistent shape of the result for broadcastint (NOTE:
    //! INCONSISTENT with the other accum_mul)
    TensorND& accum_mul(const TensorND& lhs, fp_t rhs);

    //! set this as element-wise negation of x
    TensorND& as_neg(const TensorND& x);

    //! inplace element-wise negation
    TensorND& inplace_neg() { return as_neg(*this); }

    //! set this as pow(\p x, \p exp)
    TensorND& as_pow(const TensorND& x, fp_t exp);

    TensorND pow(fp_t exp) const {
        TensorND ret;
        ret.as_pow(*this, exp);
        return ret;
    }

    //! set this as the natural logarithm of \p x
    TensorND& as_log(const TensorND& x);

    TensorND log() const {
        TensorND ret;
        ret.as_log(*this);
        return ret;
    }

    /*!
     * \brief set this as the result of matmul between \p lhs and \p rhs
     * \param accum whether to accumulate to this tensor or overwrite the
     *      content
     */
    TensorND& as_mm(const TensorND& lhs, const TensorND& rhs,
                    bool accum = false, bool trans_lhs = false,
                    bool trans_rhs = false);

    /*!
     * \brief set this as the result of batched matmul between \p lhs and \p rhs
     * \param accum whether to accumulate to this tensor or overwrite the
     *      content
     */
    TensorND& as_batched_mm(const TensorND& lhs, const TensorND& rhs,
                            bool accum = false, bool trans_lhs = false,
                            bool trans_rhs = false);

    TensorND batched_mm(const TensorND& rhs, bool trans_self = false,
                        bool trans_rhs = false) const {
        return TensorND{}.as_batched_mm(*this, rhs, false, trans_self,
                                        trans_rhs);
    }

    //! set this as the result of batched matrix transpose
    TensorND& as_batched_transpose(const TensorND& src);

    //! set this as the transpose of \p src
    TensorND& as_transpose(const TensorND& src);

    //! see as_batched_transpose()
    TensorND batched_transpose() const {
        return TensorND{}.as_batched_transpose(*this);
    }

    //! set this as the result of batched square matrix inversion
    TensorND& as_batched_matinv(const TensorND& src);

    //! see as_batched_matinv()
    TensorND batched_matinv() const {
        return TensorND{}.as_batched_matinv(*this);
    }

    //! set this as the result of batched determinant of square matrices
    TensorND& as_batched_determinant(const TensorND& src);

    //! see as_batched_determinant
    TensorND batched_determinant() const {
        return TensorND{}.as_batched_determinant(*this);
    }

    //! set this as the result of batched cofactor matrices
    TensorND& as_batched_cofactor(const TensorND& src);

    //! see as_batched_cofactor()
    TensorND batched_cofactor() const {
        return TensorND{}.as_batched_cofactor(*this);
    }

    /*!
     * \brief compute batch matrix-matrix multiplication, where the items in the
     *      left matrix are vectors
     *
     * \p lhs must have shape (batch, m * k, p) and \p rhs must have shape
     * (batch, k, n). Matrix multiplication is performed on the middle two
     * dimensions. Output shape is (batch, m * n, p).
     *
     * Einstein notation: out[b, (m, n), p] = l[b, (m, k), p] * r[b, k, n]
     *
     * \param accum whether to accumulate the result into this
     */
    TensorND& as_batched_mm_vecitem_left(const TensorND& lhs,
                                         const TensorND& rhs,
                                         bool accum = false);

    /*!
     * \brief compute a varient singular value decomposition (which we call
     *      SVD-W) in the batched fasion
     *
     * this = u @ s @ u' @ w, where both u and w are unitary. In the context of
     * conventional SVD, w = uv', and equivalently v' = u'w. We also have w'w=I.
     * The input matrix must be square.
     *
     * We compute this form of SVD because the jacobians of u and v are not
     * well-defined in the degenerated case (u and v corresponding to identical
     * singular values can be rotated by arbitrary unitary matrices), but w=uv'
     * is well defined.
     *
     * \param require_rotation whether w should be a proper rotation. If set to
     *      true, a singular value and the corresponding columns in u and v
     *      might be negated to ensure that det(w) = 1. The implementation
     *      tries to avoid negating repeated singular values equal to -1 as much
     *      as possible to improve grad numerical stability.
     */
    const TensorND& compute_batched_svd_w(TensorND& u, TensorND& s, TensorND& w,
                                          bool require_rotation = false) const;

    //! return (U, S, W). See compute_batched_svd_w
    std::array<TensorND, 3> batched_svd_w(bool require_rotation = false) const {
        TensorND u, s, w;
        compute_batched_svd_w(u, s, w, require_rotation);
        return {u, s, w};
    }

    //! expand vector \p src into a diagnoal matrix
    TensorND& as_diag(const TensorND& src);

    //! expand batched vectors \p src into batched diagnoal matrices
    TensorND& as_batched_diag(const TensorND& src);

    //! expand batched scalar \p src into diagnoal matrices
    TensorND& as_batched_diag_scalar(const TensorND& src, size_t dim);

    //! as the batched traces of square matrices; output dim [batch, 1]
    TensorND& as_batched_trace(const TensorND& src);

    /*!
     * \brief compute the sum of elements along a single axis
     * \param axis the reduction axis; use -1 for batch flattened tensor (result
     *      shape is [batch, 1]), and -2 for reducing to scalar (result shape is
     *      [1])
     * \param keepdim whether to retain the reduced dimension (no effect when
     *      axis is -1 or -2)
     */
    TensorND& as_reduce_sum(const TensorND& src, int axis, bool keepdim);

    //! set this as \p src broadcasted along \p axis to size \p size; the
    //! original size on this axis must be 1.
    TensorND& as_broadcast(const TensorND& src, size_t axis, size_t size);

    //! see as_reduce_sum()
    TensorND reduce_sum(int axis, bool keepdim = true) const {
        return TensorND{}.as_reduce_sum(*this, axis, keepdim);
    }

    //! L2-norm (i.e., square root of sum of squares)
    fp_t norm_l2() const;

    //! squared L2-norm (i.e., sum of squares)
    fp_t squared_norm_l2() const;

    //! rooted mean squared
    fp_t norm_rms() const;

    //! compute dot product on the flattened tensors
    fp_t flat_dot(const TensorND& rhs) const;

    /*!
     * \brief assert that all items in this tensor are close to corresponding
     *      items in the other tensor.
     *
     * This function is used for debug
     *
     * \param msg error message when assertion fails
     * \throw SANMAssertionError is raised if assertion fails.
     */
    void assert_allclose(const char* msg, const TensorND& rhs,
                         fp_t eps = 1e-4) const;
};
using TensorArray = std::vector<TensorND>;

//! see TensorND::reduce_sum()
TensorShape infer_reduce_shape(const TensorShape& src, int axis, bool keepdim);

/*!
 * \brief compute the coefficient of the term with given order in the
 *      determinant of a polynomial matrix
 * \param a coefficients of the polynomial matrix; must be batched square
 *      matrices of the same shape
 * \param order the order of the term to be computed
 * \return computed scalar coefficient (shape: (batch, 1))
 */
TensorND compute_polymat_det_coeff(const TensorArray& a, size_t order);

//! structurally sparse linear transforms (without bias) for gradient computing
class StSparseLinearTrans {
public:
    enum Type {
        INVALID,   //!< uninitialized
        ELEMWISE,  //!< elemetwise multiplication by this tensor
        FULL,      //!< this shape: (m, k); x flatten shape: (k, )
    };

    StSparseLinearTrans() = default;
    StSparseLinearTrans(Type type, bool is_batched, const TensorND& coeff) {
        reset(type, is_batched, coeff);
    }

    bool valid() const { return m_type != INVALID; }

    Type type() const { return m_type; }

    /*!
     * \brief whether this transform is batched
     * If type is FULL, then batched transform has a shape of (b, m, k)
     */
    bool is_batched() const {
        sanm_assert(valid());
        return m_is_batched;
    }

    //! batch size (accessible if is_batched() is true)
    size_t batch() const {
        sanm_assert(is_batched());
        return m_batch;
    }

    //! output dimension
    size_t out_dim() const {
        sanm_assert(valid());
        return m_out_dim;
    }

    //! input dimension
    size_t inp_dim() const {
        sanm_assert(valid());
        return m_inp_dim;
    }

    //! get the underlying coefficient tensor
    const TensorND& coeff() const {
        sanm_assert(valid());
        return m_coeff;
    }

    //! reset the underlying transform; shape of \p coeff decides dimensions
    StSparseLinearTrans& reset(Type type, bool is_batched,
                               const TensorND& coeff);

    //! compute F such that F(x) = this(rhs*x). This function does not impose
    //! constraint shape of \p rhs, and it only requires the total number of
    //! elements to match
    StSparseLinearTrans compose_with_elemwise(const TensorND& rhs) const;

    //! compute F such that F(x) = this(rhs @ x). Shape of rhs must match
    StSparseLinearTrans compose_with_full(const TensorND& rhs) const;

    //! compute F such that F(x) = this(scale*x)
    StSparseLinearTrans compose_with_scaling(fp_t scale) const;

    //! compute F such that F(x) = this(x) + rhs(x), with this updated inplace
    //! to F. This can be invalid before the call.
    StSparseLinearTrans& operator+=(const StSparseLinearTrans& rhs);

    StSparseLinearTrans& accum_full(bool is_batched, const TensorND& coeff) {
        return (*this) += StSparseLinearTrans{}.reset(FULL, is_batched, coeff);
    }

    //! get the tensor as a full tensor, whose rank is 3 if is_batched() is
    //! true, and 2 otherwise
    TensorND as_full() const;

    //! apply this transform on \p x which will be flattened; the result is also
    //! flattened (or flattened)
    TensorND apply(const TensorND& x) const;

    const StSparseLinearTrans& check_batched(bool req) const {
        sanm_assert(m_is_batched == req, "require batched %d, got %d", req,
                    m_is_batched);
        return *this;
    }

private:
    Type m_type = INVALID;
    bool m_is_batched = false;
    size_t m_batch = 0, m_out_dim = 0, m_inp_dim = 0;
    TensorND m_coeff;
};

int get_num_threads();
void set_num_threads(int nr);

}  // namespace sanm
