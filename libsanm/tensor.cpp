/**
 * \file libsanm/tensor.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/tensor_impl_helper.h"

#include <mkl_service.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

using namespace sanm;

/* ======================= global ======================= */

namespace {
int nr_threads = 1;
__attribute__((constructor)) void eigen_setup() {
    Eigen::initParallel();
    Eigen::internal::set_is_malloc_allowed(false);
    // disable internal auto parallelism
    Eigen::setNbThreads(1);
    mkl_set_num_threads(1);
    // use one thread by default
    set_num_threads(1);
}

void prepare_inplace(TensorND* dst, const TensorND& src) {
    if (dst == &src) {
        dst->rwptr();
    } else {
        dst->set_shape({src.shape()});
    }
}

void check_not_inplace(TensorND* dst, const TensorND& src) {
    sanm_assert(dst != &src, "inplace not allowed");
}

template <size_t n>
void bcast_kern_static(fp_t* dst, const fp_t* src, size_t m, size_t) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            dst[i * n + j] = src[j];
        }
    }
}

void bcast_kern_dynamic(fp_t* dst, const fp_t* src, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        memcpy(dst + i * n, src, sizeof(fp_t) * n);
    }
}

typedef void (*bcast_kern_t)(fp_t*, const fp_t*, size_t, size_t);

bcast_kern_t bcast_kern_dispatch(size_t n) {
    sanm_assert(n > 0);
    switch (n) {
#define ON(x) \
    case x:   \
        return bcast_kern_static<x>
        ON(1);
        ON(2);
        ON(3);
        ON(4);
        ON(5);
        ON(6);
        ON(7);
        ON(8);
#undef ON
    }
    return bcast_kern_dynamic;
}

}  // anonymous namespace

std::atomic_size_t ScopedAllowMalloc::sm_stack_depth{0};

int sanm::get_num_threads() {
    return nr_threads;
}

void sanm::set_num_threads(int nr) {
    sanm_assert(nr >= 1);
    nr_threads = nr;
    omp_set_num_threads(nr);  // sparse solver may use omp threads
}

TensorShape sanm::infer_reduce_shape(const TensorShape& src, int axis,
                                     bool keepdim) {
    if (axis == -1) {
        return {src.dim[0], 1};
    }

    sanm_assert(axis >= 0 && axis < static_cast<int>(src.rank),
                "invalid reduce axis: %d (shape %s)", axis, src.str().c_str());

    TensorShape out_shape = src;
    if (keepdim) {
        out_shape.dim[axis] = 1;
    } else {
        out_shape.rank -= 1;
        for (size_t i = axis; i < out_shape.rank; ++i) {
            out_shape.dim[i] = out_shape.dim[i + 1];
        }
    }

    return out_shape;
}

/* ======================= TensorShape ======================= */

TensorShape::TensorShape(std::initializer_list<size_t> sizes) {
    sanm_assert(sizes.size() <= MAX_RANK);
    for (auto i : sizes) {
        sanm_assert(i <= std::numeric_limits<leastsize_t>::max());
        dim[rank++] = i;
    }
}

TensorShape TensorShape::flatten_batched() const {
    sanm_assert(rank);
    size_t sp = 1;
    for (size_t i = 1; i < rank; ++i) {
        sp *= dim[i];
    }
    return {dim[0], sp};
}

TensorShape TensorShape::flatten_as_vec() const {
    sanm_assert(rank);
    size_t sp = 1;
    for (size_t i = 0; i < rank; ++i) {
        sp *= dim[i];
    }
    return {sp};
}

size_t TensorShape::total_nr_elems() const {
    if (rank == 0) {
        return 0;
    }
    size_t ret = 1;
    for (size_t i = 0; i < rank; ++i) {
        ret *= dim[i];
    }
    return ret;
}

size_t TensorShape::total_nr_elems_per_batch() const {
    if (rank == 0) {
        return 0;
    }
    size_t ret = 1;
    for (size_t i = 1; i < rank; ++i) {
        ret *= dim[i];
    }
    return ret;
}

bool TensorShape::operator==(const TensorShape& rhs) const {
    static_assert(MAX_RANK == 3);
    if (rank != rhs.rank) {
        return false;
    }
    switch (rank) {
        case 0:
            return true;
        case 3:
            if (dim[2] != rhs.dim[2]) {
                return false;
            }
            [[fallthrough]];
        case 2:
            if (dim[1] != rhs.dim[1]) {
                return false;
            }
            [[fallthrough]];
        case 1:
            return dim[0] == rhs.dim[0];
        default:
            throw SANMError{"invalid rank"};
    }
}

TensorShape TensorShape::with_batch_replaced(size_t batch) const {
    sanm_assert(rank);
    TensorShape ret = *this;
    ret.dim[0] = batch;
    return ret;
}

TensorShape TensorShape::add_axis(size_t axis) const {
    sanm_assert(rank < MAX_RANK && axis <= rank, "rank=%zu axis=%zu",
                static_cast<size_t>(rank), axis);
    TensorShape ret;
    ret.rank = rank + 1;
    for (size_t i = 0; i < axis; ++i) {
        ret.dim[i] = dim[i];
    }
    ret.dim[axis] = 1;
    for (size_t i = axis + 1; i < ret.rank; ++i) {
        ret.dim[i] = dim[i - 1];
    }
    return ret;
}

TensorShape TensorShape::remove_axis(size_t axis) const {
    sanm_assert(rank >= 2 && axis < rank && dim[axis] == 1,
                "can not remove axis %zu from shape %s", axis, str().c_str());
    TensorShape ret;
    ret.rank = rank - 1;
    for (size_t i = 0; i < axis; ++i) {
        ret.dim[i] = dim[i];
    }
    for (size_t i = axis + 1; i < rank; ++i) {
        ret.dim[i - 1] = dim[i];
    }
    return ret;
}

std::string TensorShape::str() const {
    std::string ret{"{"};
    for (size_t i = 0; i < rank; ++i) {
        if (i) {
            ret += ",";
        }
        ret += std::to_string(dim[i]);
    }
    ret += "}";
    return ret;
}

/* ======================= TensorStorage ======================= */

TensorStorage::~TensorStorage() {
    if (m_data) {
        ::free(m_data);
        m_size = 0;
        m_data = nullptr;
    }
}

void TensorStorage::ensure_size(size_t size) {
    if (size <= m_size) {
        return;
    }
    size_t next_size =
            std::min(std::max(m_size * 2, size),
                     size + (1 << 20));  // max pre-alloc of 1MB elements
    size_t next_size_b = next_size * sizeof(fp_t);
    void* next_data;
    if (auto err = ::posix_memalign(&next_data, EIGEN_IDEAL_MAX_ALIGN_BYTES,
                                    next_size_b)) {
        throw SANMError{
                ssprintf("malloc failed: size=%zu err=%d", next_size_b, err)};
    }
    ::free(m_data);
    m_data = static_cast<fp_t*>(next_data);
    m_size = next_size;
}

template <TensorStorage::Constant VAL>
const std::shared_ptr<const TensorStorage>& TensorStorage::constant(
        size_t size) {
    static std::shared_ptr<const TensorStorage> storage =
            std::make_shared<TensorStorage>();
    static std::atomic_size_t storage_size{0};
    static std::mutex mutex;
    if (size && size > storage_size.load()) {
        std::lock_guard<std::mutex> lock{mutex};
        auto p = const_cast<TensorStorage*>(storage.get());
        if (size > p->size()) {
            p->ensure_size(size);
            if constexpr (VAL == Constant::ZERO) {
                memset(p->ptr(), 0, sizeof(fp_t) * p->size());
            } else if constexpr (VAL == Constant::ONE) {
                std::fill_n(p->ptr(), p->size(), 1._fp);
            } else {
                sanm_assert(0, "impossible");
            }
            storage_size.store(p->size());
        }
    }
    return storage;
}

/* ======================= TensorND ======================= */

TensorND& TensorND::set_shape(const TensorShape& shape) {
    if (!m_storage || m_storage.use_count() > 1) {
        m_storage = std::make_shared<TensorStorage>();
    }
    m_storage->ensure_size(shape.total_nr_elems());
    m_shape = shape;
    return *this;
}

TensorND& TensorND::incr_sub_batch(const TensorND& src, size_t dbatch,
                                   size_t sbatch, size_t size, fp_t alpha,
                                   fp_t beta) {
    check_not_inplace(this, src);
    sanm_assert(sbatch + size <= src.shape(0));
    TensorShape dst_shape;
    if (!dbatch && empty()) {
        dst_shape = src.shape().with_batch_replaced(size);
        set_shape(dst_shape);
    } else {
        dst_shape = src.shape().with_batch_replaced(shape(0));
        sanm_assert(m_shape == dst_shape);
        sanm_assert(size <= dst_shape.dim[0]);
    }

    size_t elem = 1;
    for (size_t i = 1; i < dst_shape.rank; ++i) {
        elem *= dst_shape.dim[i];
    }

    fp_t* dptr = rwptr() + dbatch * elem;
    const fp_t* sptr = src.ptr() + sbatch * elem;
    size_t tot_size = size * elem;
    if (alpha == 0 && beta == 1) {
        memcpy(dptr, sptr, sizeof(fp_t) * tot_size);
    } else {
        EigenVec dv(dptr, tot_size);
        EigenCVec sv(sptr, tot_size);
        dv.array() = dv.array() * alpha + sv.array() * beta;
    }
    return *this;
}

TensorND& TensorND::as_reshaped(const TensorND& src, const TensorShape& shape) {
    sanm_assert(src.m_shape.total_nr_elems() == shape.total_nr_elems(),
                "can not reshape from %s to %s", src.m_shape.str().c_str(),
                shape.str().c_str());
    if (this != &src) {
        m_storage = src.m_storage;
    }
    m_shape = shape;
    return *this;
}

TensorND TensorND::take_sub(const TensorShape& shape) const {
    sanm_assert(shape.total_nr_elems() <= m_shape.total_nr_elems(),
                "can not take subtensor of shape %s from %s",
                shape.str().c_str(), m_shape.str().c_str());
    TensorND ret;
    ret.m_shape = shape;
    ret.m_storage = m_storage;
    return ret;
}

TensorND& TensorND::fill_with_inplace(fp_t value) {
    size_t num = shape().total_nr_elems();
    using C = TensorStorage::Constant;
    if (value == 0._fp) {
        m_storage = std::const_pointer_cast<TensorStorage>(
                TensorStorage::constant<C::ZERO>(num));
        return *this;
    }
    if (value == 1._fp) {
        m_storage = std::const_pointer_cast<TensorStorage>(
                TensorStorage::constant<C::ONE>(num));
        return *this;
    }
    std::fill_n(woptr(), num, value);
    return *this;
}

TensorND TensorND::operator*(fp_t scale) const {
    if (scale == 0. || scale == -0.) {
        return fill_with(0);
    }
    if (scale == 1) {
        return *this;
    }
    if (is_zero()) {
        return *this;
    }
    auto ret = make_same_shape();
    as_vector_w(ret) = as_vector_r(*this) * scale;
    return ret;
}

TensorND& TensorND::operator*=(fp_t scale) {
    if (scale == 0. || scale == -0.) {
        return fill_with_inplace(0);
    }
    if (is_zero()) {
        return *this;
    }
    if (scale != 1) {
        as_vector_w(*this) *= scale;
    }
    return *this;
}

TensorND& TensorND::as_pow(const TensorND& x, fp_t exp) {
    if (x.is_zero()) {
        sanm_assert(exp > 0);
        *this = x;
        return *this;
    }
    prepare_inplace(this, x);
    EigenVec dst = as_vector_w(*this);
    EigenVec src_evec = as_vector_r(x);
    auto src = src_evec.array();
    if (exp == -3.) {
        dst = src.cube().inverse();
        return *this;
    }
    if (exp == -2.) {
        dst = src.square().inverse();
        return *this;
    }
    if (exp == -1.) {
        dst = src.inverse();
        return *this;
    }
    if (exp == -.5) {
        dst = src.rsqrt();
        return *this;
    }
    if (exp == .0) {
        return fill_with_inplace(1);
    }
    if (exp == .5) {
        dst = src.sqrt();
        return *this;
    }
    if (exp == 1.) {
        *this = x;
        return *this;
    }
    if (exp == 2.) {
        dst = src.square();
        return *this;
    }
    if (exp == 3.) {
        dst = src.cube();
        return *this;
    }
    dst = src.pow(exp);
    return *this;
}

TensorND& TensorND::as_log(const TensorND& x) {
    prepare_inplace(this, x);
    EigenVec dst = as_vector_w(set_shape(x.shape()));
    dst = as_vector_r(x).array().log();
    return *this;
}

TensorND& TensorND::as_neg(const TensorND& x) {
    if (x.is_zero()) {
        *this = x;
        return *this;
    }
    prepare_inplace(this, x);
    as_vector_w(*this) = -as_vector_r(x).array();
    return *this;
}

TensorND& TensorND::as_diag(const TensorND& src) {
    check_not_inplace(this, src);
    sanm_assert(src.rank() == 1);
    size_t dim = src.shape(0);
    auto optr = this->set_shape({dim, dim}).woptr();
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    auto iptr = src.ptr();
    memset(optr, 0, sizeof(fp_t) * dim * dim);
    for (size_t i = 0; i < dim; ++i) {
        optr[i * dim + i] = iptr[i];
    }
    return *this;
}

TensorND& TensorND::as_batched_diag(const TensorND& src) {
    check_not_inplace(this, src);
    sanm_assert(src.rank() == 2);
    size_t batch = src.shape(0), dim = src.shape(1);
    auto optr = this->set_shape({batch, dim, dim}).woptr();
    auto iptr = src.ptr();
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    memset(optr, 0, sizeof(fp_t) * batch * dim * dim);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            optr[((i * dim) + j) * dim + j] = iptr[i * dim + j];
        }
    }
    return *this;
}

TensorND& TensorND::as_batched_diag_scalar(const TensorND& src, size_t dim) {
    check_not_inplace(this, src);
    auto tshp = src.shape();
    sanm_assert(tshp.is_batched_scalar(),
                "as_batched_diag_scalar expacts a scalar, got %s",
                tshp.str().c_str());
    tshp.rank = 3;
    tshp.dim[1] = tshp.dim[2] = dim;
    auto iptr = src.ptr();
    auto optr = this->set_shape(tshp).woptr();
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    memset(optr, 0, sizeof(fp_t) * tshp.total_nr_elems());
    size_t batch = tshp.dim[0];
    for (size_t i = 0; i < batch; ++i) {
        fp_t v = iptr[i];
        for (size_t j = 0; j < dim; ++j) {
            optr[(i * dim + j) * dim + j] = v;
        }
    }
    return *this;
}

TensorND& TensorND::as_batched_trace(const TensorND& src) {
    check_not_inplace(this, src);
    sanm_assert(src.rank() == 3 && src.shape(1) == src.shape(2),
                "not batched square matrices: %s", src.shape().str().c_str());
    const size_t batch = src.shape(0), dim = src.shape(1);
    auto iptr = src.ptr();
    auto optr = this->set_shape({batch, 1}).woptr();
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    for (size_t i = 0; i < batch; ++i) {
        fp_t sum = 0;
        for (size_t j = 0; j < dim; ++j) {
            sum += iptr[(i * dim + j) * dim + j];
        }
        optr[i] = sum;
    }
    return *this;
}

TensorND& TensorND::as_reduce_sum(const TensorND& src, int axis, bool keepdim) {
    check_not_inplace(this, src);
    sanm_assert(this != &src);

    set_shape(infer_reduce_shape(src.shape(), axis, keepdim));
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }

    Eigen::Index size_before = 1, size_ax = 1, size_after = 1;
    if (axis == -1) {
        size_before = src.shape(0);
        for (size_t i = 1; i < src.rank(); ++i) {
            size_ax *= src.shape().dim[i];
        }
    } else {
        size_ax = src.shape(axis);
        for (int i = 0; i < axis; ++i) {
            size_before *= src.shape().dim[i];
        }
        for (size_t i = axis + 1; i < src.rank(); ++i) {
            size_after *= src.shape().dim[i];
        }
    }

    fp_t *dptr = woptr(), *sptr = const_cast<fp_t*>(src.ptr());

    if (size_after == 1) {
        EigenVec mdst{dptr, size_before};
        EigenMatDyn msrc{sptr, size_ax, size_before};
        mdst = msrc.colwise().sum().transpose();
        return *this;
    }

    for (Eigen::Index i = 0; i < size_before; ++i) {
        EigenVec mdst{dptr + i * size_after, size_after};
        EigenMatDyn msrc{sptr + i * size_ax * size_after, size_after, size_ax};
        mdst = msrc.rowwise().sum();
    }
    return *this;
}

TensorND& TensorND::as_broadcast(const TensorND& src, size_t axis,
                                 size_t size) {
    check_not_inplace(this, src);
    sanm_assert(axis < src.rank() && src.shape(axis) == 1 && size > 0);
    if (size == 1) {
        *this = src;
        return *this;
    }
    size_t size_before = 1, size_after = 1;
    for (size_t i = 0; i < axis; ++i) {
        size_before *= src.shape(i);
    }
    for (size_t i = axis + 1; i < src.rank(); ++i) {
        size_after *= src.shape(i);
    }
    TensorShape tshp = src.shape();
    tshp.dim[axis] = size;
    auto iptr = src.ptr();
    auto optr = this->set_shape(tshp).woptr();
    if (src.is_zero()) {
        return fill_with_inplace(0);
    }
    auto kern = bcast_kern_dispatch(size_after);
    for (size_t i = 0; i < size_before; ++i) {
        kern(optr + i * size * size_after, iptr + i * size_after, size,
             size_after);
    }
    return *this;
}

TensorND& TensorND::ensure_unshared_nokeep() {
    if (m_storage.use_count() > 1) {
        m_storage = std::make_shared<TensorStorage>();
        m_storage->ensure_size(m_shape.total_nr_elems());
    }
    return *this;
}

TensorND& TensorND::ensure_unshared_keep() {
    if (m_storage.use_count() > 1) {
        auto new_storage = std::make_shared<TensorStorage>();
        size_t nr = shape().total_nr_elems();
        new_storage->ensure_size(nr);
        if (is_zero()) {
            memset(new_storage->ptr(), 0, nr * sizeof(fp_t));
        } else {
            memcpy(new_storage->ptr(), m_storage->ptr(), nr * sizeof(fp_t));
        }
        m_storage.swap(new_storage);
    }
    return *this;
}

fp_t TensorND::norm_l2() const {
    if (is_zero()) {
        return 0;
    }
    return as_vector_r(*this).norm();
}

fp_t TensorND::squared_norm_l2() const {
    if (is_zero()) {
        return 0;
    }
    return as_vector_r(*this).squaredNorm();
}

fp_t TensorND::norm_rms() const {
    return std::sqrt(squared_norm_l2() / shape().total_nr_elems());
}

fp_t TensorND::flat_dot(const TensorND& rhs) const {
    sanm_assert(m_shape.total_nr_elems() == rhs.shape().total_nr_elems());
    if (is_zero() || rhs.is_zero()) {
        return 0;
    }
    return as_vector_r(*this).dot(as_vector_r(rhs));
}

void TensorND::assert_allclose(const char* msg, const TensorND& rhs,
                               fp_t eps) const {
    sanm_assert(shape() == rhs.shape(), "%s: shape mismatch: %s vs %s", msg,
                shape().str().c_str(), rhs.shape().str().c_str());
    auto p0 = this->ptr(), p1 = rhs.ptr();
    for (size_t i = 0, it = shape().total_nr_elems(); i < it; ++i) {
        fp_t v0 = p0[i], v1 = p1[i],
             maxerr = eps *
                      std::max<fp_t>(std::min(std::fabs(v0), std::fabs(v1)), 1);
        sanm_assert(std::fabs(v0 - v1) < maxerr,
                    "%s: value mismatch at index %zu on shape %s: v0=%g v1=%g "
                    "(eps=%g)",
                    msg, i, shape().str().c_str(), v0, v1, eps);
    }
}

/* ======================= StSparseLinearTrans ======================= */

StSparseLinearTrans& StSparseLinearTrans::reset(Type type, bool is_batched,
                                                const TensorND& coeff) {
    size_t expect_rank, ro, ri;
    ro = is_batched ? 1 : 0;
    switch (type) {
        case ELEMWISE:
            expect_rank = is_batched ? 2 : 1;
            ri = ro;
            break;
        case FULL:
            expect_rank = is_batched ? 3 : 2;
            ri = ro + 1;
            break;
        default:
            throw SANMError{ssprintf("invalid StSparseLinearTrans type: %d",
                                     static_cast<int>(type))};
    }
    sanm_assert(coeff.rank() == expect_rank, "expected rank=%zu, shape=%s",
                expect_rank, coeff.shape().str().c_str());
    m_type = type;
    m_is_batched = is_batched;
    m_batch = is_batched ? coeff.shape(0) : 0;
    m_out_dim = coeff.shape(ro);
    m_inp_dim = coeff.shape(ri);
    m_coeff = coeff;
    return *this;
}

StSparseLinearTrans StSparseLinearTrans::compose_with_elemwise(
        const TensorND& rhs) const {
    sanm_assert(valid());
    TensorND rhs_f;
    if (m_is_batched) {
        rhs_f = rhs.flatten_batched();
        sanm_assert(rhs_f.shape(0) == m_batch && rhs_f.shape(1) == m_inp_dim,
                    "shape mismatch: batch=%zu idim=%zu rhs=%s", m_batch,
                    m_inp_dim, rhs.shape().str().c_str());
        if (m_type == FULL) {
            rhs_f.reshape_inplace({m_batch, 1, m_inp_dim});
        }
    } else {
        rhs_f = rhs.flatten_as_vec();
        sanm_assert(rhs_f.shape(0) == m_inp_dim,
                    "shape mismatch: idim=%zu rhs=%s", m_inp_dim,
                    rhs.shape().str().c_str());
        if (m_type == FULL) {
            rhs_f.reshape_inplace({1, m_inp_dim});
        }
    }
    return {m_type, m_is_batched, m_coeff * rhs_f};
}

StSparseLinearTrans StSparseLinearTrans::compose_with_full(
        const TensorND& rhs) const {
    sanm_assert(valid());
    auto lhs = as_full();
    sanm_assert(rhs.rank() == lhs.rank(), "shape mismatch: %s",
                rhs.shape().str().c_str());
    TensorND ret;
    if (is_batched()) {
        ret.as_batched_mm(lhs, rhs);
    } else {
        ret.as_mm(lhs, rhs);
    }
    return {FULL, m_is_batched, ret};
}

StSparseLinearTrans StSparseLinearTrans::compose_with_scaling(
        fp_t scale) const {
    sanm_assert(valid());
    return {m_type, m_is_batched, m_coeff * scale};
}

StSparseLinearTrans& StSparseLinearTrans::operator+=(
        const StSparseLinearTrans& rhs) {
    sanm_assert(rhs.valid());
    if (!valid()) {
        *this = rhs;
        return *this;
    }
    sanm_assert(m_is_batched == rhs.m_is_batched,
                "batch setting mismatches in add-assign: %d vs %d",
                m_is_batched, rhs.m_is_batched);
    sanm_assert(m_batch == rhs.m_batch && m_inp_dim == rhs.m_inp_dim &&
                m_out_dim == rhs.m_out_dim);
    if (m_type == rhs.m_type) {
        m_coeff += rhs.m_coeff;
    } else {
        m_coeff = as_full();
        m_coeff += rhs.as_full();
        m_type = FULL;
    }
    return *this;
}

TensorND StSparseLinearTrans::as_full() const {
    sanm_assert(valid());
    if (is_batched()) {
        if (m_coeff.rank() == 3) {
            return m_coeff;
        }
        return TensorND{}.as_batched_diag(m_coeff);
    }
    if (m_coeff.rank() == 2) {
        return m_coeff;
    }
    return TensorND{}.as_diag(m_coeff);
}

TensorND StSparseLinearTrans::apply(const TensorND& x) const {
    sanm_assert(valid());
    TensorND xflat;
    if (is_batched()) {
        xflat = x.flatten_batched();
        sanm_assert(xflat.shape(0) == m_batch && xflat.shape(1) == m_inp_dim,
                    "shape mismatch: batch=%zu inp_dim=%zu x=%s", m_batch,
                    m_inp_dim, x.shape().str().c_str());
        if (m_type == FULL) {
            xflat.reshape_inplace({xflat.shape(0), xflat.shape(1), 1});
            return TensorND{}
                    .as_batched_mm(m_coeff, xflat)
                    .reshape({m_batch, m_out_dim});
        }
    } else {
        xflat = x.flatten_as_vec();
        sanm_assert(xflat.shape(0) == m_inp_dim,
                    "shape mismatch: inp_dim=%zu x=%s", m_inp_dim,
                    x.shape().str().c_str());
        if (m_type == FULL) {
            xflat.reshape_inplace({xflat.shape(0), 1});
            return TensorND{}.as_mm(m_coeff, xflat).reshape({m_out_dim});
        }
    }
    sanm_assert(m_type == ELEMWISE);
    return m_coeff * xflat;
}
