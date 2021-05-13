/**
 * \file libsanm/analytic_unary.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// unary analytic functions applied in an element-wise manner
#pragma once

#include "libsanm/tensor.h"

namespace sanm {

class UnaryAnalyticTrait;
using UnaryAnalyticTraitPtr = std::shared_ptr<const UnaryAnalyticTrait>;

//! specific information of a function
class UnaryAnalyticTrait : public NonCopyable {
public:
    //! see prop_taylor_coeff()
    class TaylorCoeffUserData {
    public:
        virtual ~TaylorCoeffUserData() = default;
    };
    using TaylorCoeffUserDataPtr = std::unique_ptr<TaylorCoeffUserData>;

    //! evaluate the function
    virtual void eval(TensorND& dst, const TensorND& src) const = 0;

    //! compute the first order derivative of the function
    virtual void eval_derivative(TensorND& dst, const TensorND& src) const = 0;

    /*!
     * \brief Propogate the next Taylor expansion coefficient
     *
     * Compute the coefficient of \f$a^k\f$ in the expansion of
     * \f$f(\sum_{i=0}^{k-1} x_ia^i)\f$
     *
     * \param f the computed coefficients up to order \f$k-1\f$
     * \param x the coefficients of \f$x\f$ up to order \f$k-1\f$
     * \param [in,out] user_data_p extra user data which can be used by the
     *      implementation to accelerate coefficient propagation for
     *      consecutively extended \p f and \p x. If this pointer is not
     *      nullptr, then the caller intends consecutive computation of
     *      coefficients, and the implementation is allowed to modify the
     *      pointer to store extra user data.
     */
    void prop_taylor_coeff(TensorND& dst, const TensorArray& f,
                           const TensorArray& x,
                           TaylorCoeffUserDataPtr* user_data_p) const;

    TensorND eval(const TensorND& x) const {
        TensorND y;
        eval(y, x);
        return y;
    }

    //! make an UnaryAnalyticTrait for the natural logarithm function
    static UnaryAnalyticTraitPtr make_log();

    //! make an UnaryAnalyticTrait for the power function with a constant
    //! exponent
    static UnaryAnalyticTraitPtr make_pow(fp_t exp);

protected:
    ~UnaryAnalyticTrait() = default;

    //! check alias, reshape, and return write ptr
    static fp_t* setup_dst(TensorND& dst, const TensorND& src);

    //! implement prop_taylor_coeff(), with inputs having been checked and
    //! x.size() is at least 2, and \p dst already cleared
    virtual void do_prop_taylor_coeff(
            TensorND& dst, const TensorArray& f, const TensorArray& x,
            TaylorCoeffUserDataPtr* user_data_p) const = 0;
};

}  // namespace sanm
