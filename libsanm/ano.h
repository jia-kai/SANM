/**
 * \file libsanm/ano.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

//! asymptotic numerical optimizer
#pragma once

#include "libsanm/symbolic.h"
#include "libsanm/unary_polynomial.h"

namespace sanm {
/*!
 * \brief minimize f(x), given an initial value x0
 *
 * Similar to ANM that solves full-rank nonlinear systems, this class works by
 * analytically approximating \f$t(a)\f$ and \f$x(a)\f$ with a path parameter
 * \f$a\f$, such that \f$t(a) = f(x(a))\f$.
 */
class ANOMinimizer {
public:
    struct HyperParam {
        int order = 8;     //!< polynomial expansion order
        fp_t maxr = 1e-4;  //!< max residual to decide how far to go in one iter
        fp_t max_loss_diff = 1e-3;  //!< larger than #maxr for sanity check
        inline HyperParam() noexcept;
    };

    //! internal statistics
    struct Stat {
        fp_t a_bound;  //!< range of convergence
        fp_t a_m;      //!< the minimizer (i.e., \f$a\f$ for the next iteration)
        fp_t loss_diff;  //!< \f$|f(x(a)) - t(a)|\f$
    };

    class CoeffSolver;

    ANOMinimizer(symbolic::VarNode* loss, const TensorValueMap& x0,
                 std::unique_ptr<CoeffSolver> coeff_solver,
                 const HyperParam& hyper_param = {});
    ~ANOMinimizer();

    //! compute the L2-norm of the gradient
    fp_t grad_l2() const { return m_grad_flat.norm_l2(); }

    //! current loss value (i.e., \f$t(0)\f$).
    fp_t loss() const { return m_t_coeffs[0]; }

    //! get the value of the unknowns
    TensorValueMap get_x() const { return unpack_x_coeffs(m_x_coeffs[0]); }

    //! recompute the approximation at the range of convergence
    Stat update_approx();

private:
    const HyperParam m_hyper_param;
    const std::unique_ptr<CoeffSolver> m_coeff_solver;
    symbolic::VarNode* const m_loss_var;
    const fp_t m_max_a_bound;  //!< max m_a_bound to ensure numerical stability
    Maybe<symbolic::TaylorCoeffProp> m_taylor_prop;
    //! vars corresponding to parts in #m_x_coeffs[0]
    std::vector<std::pair<symbolic::VarNode*, TensorShape>> m_x0_vars;
    TensorArray m_x_coeffs;        //!< expansion of x(a)
    std::vector<fp_t> m_t_coeffs;  //!< expansion of t(a)
    TensorND m_grad_flat;
    size_t m_iter = 0;

    //! given m_x_coeffs[0] and m_t_coeffs[0], solve other terms
    void solve_expansion_coeffs();

    //! compute a bound from the expansion
    fp_t estimate_valid_range() const;

    //! initialize #m_taylor_prop and the gradients, without computing further
    //! terms
    void init_grad(const TensorValueMap& x0);

    TensorValueMap unpack_x_coeffs(const TensorND& xflat) const;
};

/*!
 * \brief solve the expansion coefficients \f$x_i\f$ and \f$t_i\f$.
 *
 * We have a few different strategies for solving the coefficients for this
 * under-constrained system.
 */
class ANOMinimizer::CoeffSolver {
    class GradScale;
    class Random;
    class GDApprox;

protected:
    //! solve the coefficient by scaling \p r to construct a "good" path
    //! parameterization
    static std::pair<TensorND, fp_t> solve_with_scale(
            TensorND r, const TensorND& grad, size_t order, fp_t b,
            const TensorArray& xprev, unary_polynomial::coeff_t tprev);

public:
    virtual ~CoeffSolver() = default;

    /*!
     * \brief initialize the vector \f$k\f$ (see solve()), which is the grad of
     *      loss
     *
     * This function can be called multiple types. The internal state should be
     * re-initialized after each call.
     */
    virtual void init(size_t iter, const TensorND& grad) = 0;

    /*!
     * \brief solve \f$(x_i, t_i)\f$ such that \f$k^T x_i + t_i + b = 0\f$
     * \param order the value of \f$i\f$. It is guaranteed to be consecutive
     *      integers starting from 1.
     * \param xprev previous values of x
     * \param tprev previous values of t
     */
    virtual std::pair<TensorND, fp_t> solve(
            size_t order, fp_t b, const TensorArray& xprev,
            unary_polynomial::coeff_t tprev) = 0;

    //! all of \f$x_i\f$ is a multiple of the gradient, subject to the ANM
    //! constraints
    static std::unique_ptr<CoeffSolver> make_grad_scale();

    /*!
     * \brief use random vectors for the direction of higher order terms
     *
     * The angle between the vecotr and the gradient follows a uniform
     * distribution between [0, \p max_angle].
     */
    static std::unique_ptr<CoeffSolver> make_random(fp_t max_angle,
                                                    size_t seed = 23);

    //! approximate gradient descent: t(a) = f(x0 + g * a)
    static std::unique_ptr<CoeffSolver> make_gd_approx(fp_t momentum);
};

ANOMinimizer::HyperParam::HyperParam() noexcept = default;
}  // namespace sanm
