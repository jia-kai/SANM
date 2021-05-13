/**
 * \file libsanm/pade.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "tensor.h"

#include <span>

namespace sanm {

/*!
 * \brief Pade approximation for a vector polynomial
 *
 * see A Critical Review of Asymptotic Numerical Methods, Appendix 1
 *
 * \param anm_cond whether the coefficients satisfy ANM condition:
 *      xs[i].dot(xs[1]) == (i == 1)
 */
class PadeApproximation {
    const bool m_sanity_check;
    std::span<const TensorND> m_xs;
    std::vector<fp_t> m_d;     //! solved coefficient of d
    std::vector<fp_t> m_d_lo;  //! solved coefficient of d with lower order
    std::vector<fp_t> m_t_nume_coeffs;  //!< coefficinents of t in the numerator

    fp_t m_t0 = 0, m_t_max = 0, m_t_max_a = 0;

    //! eval the numerator, without bias term and the outside a
    TensorND eval_nume(fp_t a) const {
        return eval_nume(a, m_d.data(), m_xs.size() - 2);
    }

    TensorND eval_nume(fp_t a, const fp_t* d, int n) const;

    fp_t eval_t(fp_t a) const;

public:
    explicit PadeApproximation(std::span<const TensorND> xs, bool anm_cond,
                               bool sanity_check);

    //! check whether range of convergence is larger than \p start, and setup
    //! m_t_max_a
    bool estimate_valid_range(fp_t start, fp_t eps, fp_t limit = 0);

    //! t coeffs are assumed to be the last element in \p xs
    fp_t get_t_max() const { return m_t_max; }

    //! value of a corresponding to tmax
    fp_t get_t_max_a() const { return m_t_max_a; }

    fp_t solve_a(fp_t t) const;

    std::pair<TensorND, fp_t> eval(fp_t a) const;

    //! eval concatenated x and t
    TensorND eval_xt(fp_t a) const;
};

}  // namespace sanm
