/**
 * \file libsanm/tensor_polymat.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/stl.h"
#include "libsanm/tensor_impl_helper.h"

#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <mutex>
#include <span>

using namespace sanm;

namespace {

using cfp_t = std::complex<fp_t>;

/*!
 * \brief compute the DFT of given coeffs using the FFT algorithm
 *
 * Evaluate the polynomial defined by the \p coeffs at the points \f$\omega_n^0,
 * \ldots, \omega_n^{n-1}\f$ where \f$n\f$ is \p nr_term and must be a power of
 * two.
 *
 * \return the real part and the imaginary part
 */
std::pair<TensorArray, TensorArray> fft(const TensorArray& coeffs,
                                        size_t nr_term) {
    sanm_assert(!(nr_term & (nr_term - 1)));
    if (coeffs.size() == 1) {
        return {TensorArray(nr_term, coeffs[0]),
                TensorArray(nr_term, coeffs[0].fill_with(0))};
    }
    if (nr_term == 1) {
        TensorND sum = coeffs[0];
        for (size_t i = 1; i < coeffs.size(); ++i) {
            sum += coeffs[i];
        }
        return {{sum}, {sum.fill_with(0)}};
    }

    TensorArray coeffs_even, coeffs_odd;
    coeffs_even.reserve((coeffs.size() + 1) / 2);
    coeffs_odd.reserve(coeffs.size() / 2);
    for (size_t i = 0; i < coeffs.size(); i += 2) {
        coeffs_even.emplace_back(coeffs[i]);
        if (i + 1 < coeffs.size()) {
            coeffs_odd.emplace_back(coeffs[i + 1]);
        }
    }

    auto result_even = fft(coeffs_even, nr_term / 2),
         result_odd = fft(coeffs_odd, nr_term / 2);

    TensorArray ret_real(nr_term), ret_imag(nr_term);
    for (size_t si = 0; si < nr_term / 2; ++si) {
        const TensorND& a_real = result_even.first[si];
        const TensorND& a_imag = result_even.second[si];
        const TensorND& b_real = result_odd.first[si];
        const TensorND& b_imag = result_odd.second[si];

        for (size_t i : {si, si + nr_term / 2}) {
            TensorND& y_real = ret_real[i];
            TensorND& y_imag = ret_imag[i];

            fp_t angle = fp_t(i) * (M_PI * 2) / fp_t(nr_term),
                 x_real = std::cos(angle), x_imag = std::sin(angle);

            // y = a + x * b
            y_real.set_shape(coeffs[0].shape());
            y_imag.set_shape(coeffs[0].shape());
            as_vector_w(y_real) = as_vector_r(a_real) +
                                  as_vector_r(b_real) * x_real -
                                  as_vector_r(b_imag) * x_imag;
            as_vector_w(y_imag) = as_vector_r(a_imag) +
                                  as_vector_r(b_imag) * x_real +
                                  as_vector_r(b_real) * x_imag;
        }
    }
    return {std::move(ret_real), std::move(ret_imag)};
}

size_t next_pow2(size_t x) {
    size_t y = 1;
    while (y < x) {
        y <<= 1;
    }
    return y;
}

TensorND compute_polymat_det_coeff_with_fft(const TensorArray& coeffs,
                                            size_t nr_term_pow2,
                                            size_t target_order) {
    std::pair<TensorArray, TensorArray> polymat_dft;
    {
        SANM_SCOPED_PROFILER("polymat_det_fft");
        polymat_dft = fft(coeffs, nr_term_pow2);
    }

    const size_t batch = coeffs[0].shape(0);
    const Eigen::Index mdim = coeffs[0].shape(1);
    ScopedAllowMalloc allow_malloc;
    Eigen::Matrix<cfp_t, Eigen::Dynamic, Eigen::Dynamic> eigmat(mdim, mdim);

    TensorND ret{TensorShape{batch, 1}};
    auto ret_ptr = ret.woptr();
    for (size_t ib = 0; ib < batch; ++ib) {
        // do the inverse dft to solve the target coefficient
        cfp_t accum = 0;
        for (size_t i = 0; i < nr_term_pow2; ++i) {
            EigenMatDyn mreal{const_cast<fp_t*>(polymat_dft.first[i].ptr()) +
                                      ib * mdim * mdim,
                              mdim, mdim},
                    mimag{const_cast<fp_t*>(polymat_dft.second[i].ptr()) +
                                  ib * mdim * mdim,
                          mdim, mdim};
            eigmat.real() = mreal;
            eigmat.imag() = mimag;
            cfp_t dfti = eigmat.determinant();
            fp_t angle =
                    -(M_PI * 2) * fp_t(i * target_order) / fp_t(nr_term_pow2);
            accum += dfti * cfp_t{std::cos(angle), std::sin(angle)};
        }
        accum /= fp_t(nr_term_pow2);
        sanm_assert(std::fabs(accum.imag()) <
                            1e-4 * std::max<fp_t>(1, std::fabs(accum.real())),
                    "IDFT not real: real=%g imag=%g", accum.real(),
                    accum.imag());
        ret_ptr[ib] = accum.real();
    }

    return ret;
}

TensorArray transpose_coeffs(const TensorArray& coeffs) {
    TensorArray ret;
    ret.resize(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) {
        TensorND& dst = ret[i];
        const TensorND& src = coeffs[i];
        sanm_assert(src.rank() == 3);
        size_t n = src.shape(0), m0 = src.shape(1), m1 = src.shape(2);
        dst.set_shape({m0, m1, n});
        EigenMatDyn mdst{dst.woptr(), static_cast<Eigen::Index>(n),
                         static_cast<Eigen::Index>(m0 * m1)},
                msrc{const_cast<fp_t*>(src.ptr()),
                     static_cast<Eigen::Index>(m0 * m1),
                     static_cast<Eigen::Index>(n)};
        mdst = msrc.transpose();
    }
    return ret;
}

using EigenVecArr = std::span<EigenVec>;

void conv(EigenVecArr dst, EigenVecArr x, EigenVecArr y) {
    for (EigenVec& i : dst) {
        i.setZero();
    }
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size() && i + j < dst.size(); ++j) {
            dst[i + j].array() += x[i].array() * y[j].array();
        }
    }
}

void conv_k(size_t k, EigenVec& dst, EigenVecArr x, EigenVecArr y) {
    bool first = true;
    for (size_t i = std::max(0, int(k) + 1 - int(y.size()));
         i < x.size() && i <= k; ++i) {
        if (first) {
            dst = x[i].array() * y[k - i].array();
            first = false;
        } else {
            dst.array() += x[i].array() * y[k - i].array();
        }
    }
    if (first) {
        dst.setZero();
    }
}

class EigenVecArrStorage : public ObjArray<EigenVec> {
public:
    explicit EigenVecArrStorage(size_t size)
            : ObjArray<EigenVec>(size, nullptr, 0, 1) {}
};

/*!
 * \brief Compute a single term in the expansion of determinant
 *
 * The result is the product of mat[i][row_indices[i]]
 * Result is negated if row_indices[0]<0 or row_indices[1]<0
 *
 * \param coeffs_trans transposed coefficients in (m, m, batch) shape
 * \param k target term order
 */
class DetSingleTermCompute {
    TensorArray m_coeffs_trans;
    const size_t m_batch;
    const size_t m_k;
    std::unique_ptr<fp_t[]> m_buf_storage;
    TensorND m_ret;
    EigenVecArrStorage m_buf0, m_buf1, m_opr0, m_opr1;

public:
    DetSingleTermCompute(const TensorArray& coeffs, size_t k)
            : m_coeffs_trans{transpose_coeffs(coeffs)},
              m_batch{m_coeffs_trans[0].shape(2)},
              m_k{k},
              m_buf_storage{new fp_t[2 * (k + 1) * m_batch]},
              m_ret{TensorShape{m_batch, 1}},
              m_buf0{k + 1},
              m_buf1{k + 1},
              m_opr0{coeffs.size()},
              m_opr1{coeffs.size()} {
        auto ptr = m_buf_storage.get();
        for (size_t i = 0; i <= k; ++i) {
            reset(m_buf0[i], ptr + (i * 2) * m_batch, m_batch);
            reset(m_buf1[i], ptr + (i * 2 + 1) * m_batch, m_batch);
        }
    }

    TensorND operator()(std::span<const int> row_indices) {
        auto extract = [&row_indices, this, msize = m_coeffs_trans[0].shape(1)](
                               EigenVecArr dst, int r) {
            for (size_t i = 0; i < m_coeffs_trans.size(); ++i) {
                size_t c = std::abs(row_indices[r]);
                reset(dst[i],
                      m_coeffs_trans[i].ptr() + (r * msize + c) * m_batch,
                      m_batch);
            }
        };

        auto conv_ret = [this, &row_indices](EigenVecArr x, EigenVecArr y) {
            auto ret_vec = as_vector_w(m_ret);
            conv_k(m_k, ret_vec, x, y);
            if (row_indices[0] < 0 || row_indices[1] < 0) {
                m_ret.inplace_neg();
            }
            return m_ret;
        };

        extract(m_opr0, 0);
        extract(m_opr1, 1);
        if (row_indices.size() == 2) {
            return conv_ret(m_opr0, m_opr1);
        }

        EigenVecArrStorage *prod = &m_buf0, *prod_next = &m_buf1;
        conv(*prod, m_opr0, m_opr1);

        for (size_t i = 2; i + 1 < row_indices.size(); ++i) {
            extract(m_opr0, i);
            conv(*prod_next, *prod, m_opr0);
            std::swap(prod, prod_next);
        }
        extract(m_opr0, row_indices.size() - 1);
        return conv_ret(*prod, m_opr0);
    }
};

/*!
 * \brief get the terms in the expansion of determinant
 * \return vector of size m! * m; the sign of items i*m are the sign of the
 *      terms
 */
const std::vector<int>& get_det_terms(size_t m) {
    static std::mutex mutex;
    static std::vector<std::vector<int>> results{std::vector<int>{0}};
    static auto compute = [&](int size) {
        sanm_assert(results.size() == static_cast<size_t>(size - 1));
        std::vector<int>& cur = results.emplace_back();
        const auto& prev = results[size - 2];
        for (int i = 0; i < size; ++i) {
            for (size_t j = 0; j < prev.size();) {
                size_t r0 = cur.size();
                bool neg = i % 2;
                cur.push_back(i);
                for (int jdt = 0; jdt < size - 1; ++jdt) {
                    int p = prev[j + jdt];
                    if (p < 0) {
                        neg = !neg;
                        p = -p;
                    }
                    cur.push_back(p + (p >= i));
                }
                if (neg) {
                    if (cur[r0]) {
                        cur[r0] = -cur[r0];
                    } else {
                        cur[r0 + 1] = -cur[r0 + 1];
                    }
                }
                j += size - 1;
            }
        }
#if 0
        printf("det(%d):\n", size);
        for (size_t i = 0; i < cur.size(); ++i) {
            printf("%d", cur[i]);
            if ((i + 1) % size == 0) {
                printf("\n");
            } else {
                printf(" ");
            }
        }
#endif
    };

    sanm_assert(m >= 1);
    std::lock_guard<std::mutex> mutex_lg{mutex};
    if (m - 1 < results.size()) {
        return results[m - 1];
    }
    for (size_t i = results.size() + 1; i <= m; ++i) {
        compute(i);
    }
    return results[m - 1];
}

TensorND compute_polymat_det_coeff_by_expanding(const TensorArray& coeffs,
                                                size_t target_order) {
    DetSingleTermCompute tc{coeffs, target_order};
    size_t m = coeffs[0].shape(1);
    const auto& terms = get_det_terms(m);

    TensorND ret;
    for (size_t i = 0; i < terms.size(); i += m) {
        TensorND cur = tc({terms.data() + i, m});
        if (!i) {
            ret = cur;
        } else {
            ret += cur;
        }
    }
    return ret;
}
}  // anonymous namespace

TensorND sanm::compute_polymat_det_coeff(const TensorArray& coeffs,
                                         size_t order) {
    SANM_SCOPED_PROFILER("polymat_det");
    sanm_assert(!coeffs.empty() && coeffs[0].rank() == 3 &&
                coeffs[0].shape(1) == coeffs[0].shape(2));
    for (size_t i = 1; i < coeffs.size(); ++i) {
        sanm_assert(coeffs[i].shape() == coeffs[0].shape());
    }

    const size_t batch = coeffs[0].shape(0), mdim = coeffs[0].shape(1),
                 nr_term = (coeffs.size() - 1) * mdim + 1;
    sanm_assert(mdim >= 2);

    if (order >= nr_term) {
        return TensorND{TensorShape{batch, 1}}.fill_with_inplace(0);
    }
    if (order == 0) {
        return coeffs[0].batched_determinant();
    }
    if (order == 1) {
        TensorND ret{TensorShape{batch, 1}},
                src = coeffs[0].batched_cofactor() * coeffs[1];
        EigenMatDyn smat{const_cast<fp_t*>(src.ptr()),
                         static_cast<Eigen::Index>(mdim * mdim),
                         static_cast<Eigen::Index>(batch)};
        as_vector_w(ret) = smat.colwise().sum().transpose();
        return ret;
    }

    if (mdim <= 4) {
        return compute_polymat_det_coeff_by_expanding(coeffs, order);
    }

    return compute_polymat_det_coeff_with_fft(coeffs, next_pow2(nr_term),
                                              order);
}
