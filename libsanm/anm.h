/**
 * \file libsanm/anm.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/pade.h"
#include "libsanm/symbolic.h"
#include "libsanm/unary_polynomial.h"

namespace sanm {
class SparseSolver;
class SparseLinearDesc;
using SparseLinearDescPtr = std::shared_ptr<SparseLinearDesc>;

/*!
 * \brief an abstract descriptor for describing a sparse linear mapping between
 *      tensors
 *
 * Both the source tensor and the target tensor are flattened as vectors in the
 * indexing.
 */
class SparseLinearDesc {
public:
    //! linear transformation on a single input element
    struct InputElem {
        fp_t coeff;  //!< mulplication coeff
        size_t idx;  //!< index in the source tensor

        InputElem() = default;
        InputElem(fp_t coeff_, size_t idx_) : coeff{coeff_}, idx{idx_} {}
    };

    //! describing an unbiased sparse linear transform on the inputs
    using Linear1d = std::span<const InputElem>;

    virtual ~SparseLinearDesc() = default;

    //! get the final shape of the target tensor
    virtual TensorShape out_shape() const = 0;

    //! get the expected input shape
    virtual TensorShape inp_shape() const = 0;

    //! initialize for multi-threading; the default implementation does nothing
    virtual void init_multi_thread(size_t nr_thread);

    /*!
     * \brief get the sparse linear transform on inputs to generate one output
     *
     * The implementation is allowed to modify the buffer used for previous
     * function calls on subsequent calls. The implementation must be thread
     * safe.
     *
     * \param dst_index index of the element in the target tensor
     * \param thread_id id of the caller thread, starting from zero and
     *      consecutively numbered. This can be used to create thread safe
     *      temporary storage.
     */
    virtual Linear1d get(size_t dst_index, size_t thread_id) const = 0;

    //! compute the mapped tensor from an input tensor
    TensorND apply(const TensorND& x) const;

    /*!
     * \brief create an instance the represents the identity transform
     * \param out_shape output shape, which can be omitted to use input shape
     */
    static SparseLinearDescPtr make_identity(
            const TensorShape& inp_shape,
            const Maybe<TensorShape>& out_shape = None);
};

//! general get() implementation for SparseLinearDesc with compressed storage
class SparseLinearDescCompressed : public SparseLinearDesc {
protected:
    //! all input elem flattened and concatenated together
    std::vector<InputElem> m_all_input_elem;
    //! pairs of (begin, end) in m_all_input_elem for each output index
    std::vector<std::pair<leastsize_t, leastsize_t>> m_oidx_input_elem;

public:
    Linear1d get(size_t dst_index, size_t) const override;
};

/*! Base class for ANM algorithms to drive the continuation
 *
 * Note:
 *      1. The inputs and outputs are transformed by given linear sparse
 *         transforms, so that we can have batched computation between tham. A
 *         better way is to put these transforms into the computing graph and
 *         record batch information in VarNode.
 *      2. We assume an implicit form H(x, t) = 0
 *      3. Additional constraints between \f$x\f$ and \f$t\f$ are introduced to
 *         ensure a full-rank system (Cochelin et al. [1994]).
 */
class ANMDriverHelper {
public:
    struct HyperParam {
        bool use_pade = false;     //!< whether to use Pade approximation
        bool sanity_check = true;  //!< whether to check internal solutions
        int order = 8;             //!< polynomial expansion order
        fp_t maxr = 1e-6;  //!< max residual to decide how far to go in one iter
        fp_t solution_check_tol = 1e-4;  //!< tolerance in solution sanity check

        /*!
         * add a L2 regularizer to the Taylor coefficients of x:
         * solve \f$min(|Ax-b|+\alpha|x|)\f$, which is a Tikhonov regularization
         */
        fp_t xcoeff_l2_penalty = 0;

        inline HyperParam() noexcept;
    };

    /*!
     * \brief update the polynomial expansion approximation using current upper
     *      bound of the implicit parameter \f$a\f$
     *
     * The starting point \f$(x_0,\,t_0)\f$ will be replaced by the estimated
     * value using current expansion, and the expansion will be recomputed.
     */
    void update_approx();

    //! get the upper bound of \f$t(a)\f$ for which the current expansion is
    //! accurate enough
    fp_t get_t_upper() const { return m_t_max; }

    //! solve the implicit parameter \f$a\f$ such that \f$t(a)=t\f$
    fp_t solve_a(fp_t t) const;

    //! evalulate the variables with given \f$a\f$
    std::pair<TensorND, fp_t> eval(fp_t a) const;

    //! get the coefficients [x(a); t(a)] flattened and concatenated together
    std::span<const TensorND> xt_coeffs() const { return m_xt_coeffs; }

    //! get number of iterations (i.e., number of power series evaluations)
    size_t get_nr_ieter() const { return m_iter; }

protected:
    const HyperParam m_hyper_param;
    symbolic::VarNode* const m_func_def;
    const SparseLinearDescPtr m_remap_inp, m_remap_out;
    const fp_t m_max_a_bound;  //!< max m_a_bound to ensure numerical stability
    const TensorShape m_x_shape;  //!< shape of x without t
    const size_t m_nr_unknown;    //!< size of the unknown tensor x, excluding t

    TensorND m_xt0;  //!< flattened x concatenated with t
    size_t m_iter = 0;

    ANMDriverHelper(symbolic::VarNode* f, SparseLinearDescPtr remap_inp,
                    SparseLinearDescPtr remap_out, const TensorShape& x_shape,
                    const HyperParam& hyper_param);

    ~ANMDriverHelper() = default;

    //! initialize xt0 by concat x and t
    void init_xt0(const TensorND& x, fp_t t);

    /*!
     * \brief compute the expansion coefficients starting from current m_x0 and
     *      m_t0
     *
     * The results are written to m_a_bound, m_x_coeffs and m_t_coeffs
     */
    void solve_expansion_coeffs();

    //! called by solve_expansion_coeffs() to update m_a_bound
    void estimate_valid_range();

    //! prepare xt to be fed to m_remap_inp
    virtual TensorND prepare_inp(const TensorND& xt) const = 0;

    //! get the gradient w.r.t. t
    virtual const TensorND& get_grad_t() const = 0;

    /*!
     * \brief build the coefficients of the sparse system
     * \param[out] sparse_solver the solver, which should have jacob(H, x)
     */
    virtual void build_sparse_coeff(
            SparseSolver& sparse_solver,
            symbolic::ParallelTaylorCoeffProp& coeff_prop) const = 0;

    /*!
     * \brief callback when f(x0) is computed
     * \return whether the expansion computation should continue
     */
    virtual bool on_fx0_computed(const TensorND& fx) = 0;

    fp_t get_t0() const { return m_t_coeffs[0]; }

    fp_t get_t_max_a() const { return m_t_max_a; }

    //! get concat x and t
    TensorND eval_xt(fp_t a) const;

    //! check verbose mode from env var
    static bool verbose_mode();

private:
    fp_t m_t_max = 0;         //!< max value of t(a) within the bound of a
    fp_t m_t_max_a = 0;       //!< value of a corresponding to m_t_max
    TensorArray m_xt_coeffs;  //!< expansion of [x(a); t(a)]; last element is t
    std::vector<fp_t> m_t_coeffs;  //!< t(a) extracted from m_x_coeffs

    //! setup by estimate_valid_range(); valid if pade has a larger range
    Maybe<PadeApproximation> m_pade;
};

//! use ANM to solve \f$f(x)+t*v=0\f$ for \f$x\f$
class ANMSolverVecScale : public ANMDriverHelper {
public:
    ANMSolverVecScale(symbolic::VarNode* f, SparseLinearDescPtr remap_inp,
                      SparseLinearDescPtr remap_out, TensorND x0, fp_t t0,
                      TensorND v, const HyperParam& hyper_param = {});
    virtual ~ANMSolverVecScale() = default;

protected:
    TensorND m_v;  //!< the tensor v given in the formula
    using ANMDriverHelper::ANMDriverHelper;

    TensorND prepare_inp(const TensorND& xt) const override {
        return xt.take_sub(m_x_shape);
    }

    const TensorND& get_grad_t() const override { return m_v; }

    void build_sparse_coeff(
            SparseSolver& sparse_solver,
            symbolic::ParallelTaylorCoeffProp& coeff_prop) const override;

    //! calls check_t0v_match()
    virtual bool on_fx0_computed(const TensorND& fx) override;

    //! check that fx + m_v * m_t0 == 0
    void check_t0v_match(const TensorND& fx) const;
};

//! solve the equation f(x)+y=0 with ANM
class ANMEqnSolver final : private ANMSolverVecScale {
public:
    struct HyperParam : public ANMSolverVecScale::HyperParam {
        fp_t converge_rms = 1e-5;  //!< RMS to be considered as converged

        inline HyperParam() noexcept;
    };
    ANMEqnSolver(symbolic::VarNode* f, SparseLinearDescPtr remap_inp,
                 SparseLinearDescPtr remap_out, TensorND x0, TensorND y,
                 const HyperParam& hyper_param = {});

    //! RMS of the residual vector
    fp_t residual_rms() const { return m_residual_rms; }

    //! whether the algorithm has converged
    bool converged() const { return m_converged; }

    //! compute the next iteration
    ANMEqnSolver& next_iter();

    //! current value of x
    TensorND get_x() const { return m_xt0.take_sub(m_x_shape); }

    using ANMSolverVecScale::get_nr_ieter;

private:
    const fp_t m_converge_rms;
    bool m_converged = false;
    TensorND m_eqn_y;
    fp_t m_residual_rms = 0;

    bool on_fx0_computed(const TensorND& fx) override;
};

/*!
 * \brief solve F(x, t)=F(x0, t0), where F maps R^(n+1) to R^n
 *
 * Note: F = remap_out @ f @ remap_in, while currently we require f to be a
 *      batch function
 * t increases from t0
 */
class ANMImplicitSolver final : public ANMDriverHelper {
public:
    ANMImplicitSolver(symbolic::VarNode* f, SparseLinearDescPtr remap_inp,
                      SparseLinearDescPtr remap_out, const TensorND& x0,
                      fp_t t0, const HyperParam& hyper_param = {});

    //! value of f(x0, t0)
    const TensorND& fx0() const { return m_fx0; }

protected:
    TensorND m_fx0;
    mutable TensorND m_grad_t;

    TensorND prepare_inp(const TensorND& xt) const override { return xt; }

    const TensorND& get_grad_t() const override;

    void build_sparse_coeff(
            SparseSolver& sparse_solver,
            symbolic::ParallelTaylorCoeffProp& coeff_prop) const override;

    bool on_fx0_computed(const TensorND& fx) override;
};

// define outside of the enclosing class to work around a compile error
ANMDriverHelper::HyperParam::HyperParam() noexcept = default;
ANMEqnSolver::HyperParam::HyperParam() noexcept = default;

}  // namespace sanm
