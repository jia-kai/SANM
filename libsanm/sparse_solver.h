/**
 * \file libsanm/sparse_solver.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/tensor.h"

namespace sanm {

/*!
 * \brief solve Ax = b for a sparse full-rank A
 *
 * Call add_constraint() repeatedly, then prepare(), and finally solve().
 */
class SparseSolver {
    class Impl;
    std::unique_ptr<Impl> m_pimpl;

public:
    explicit SparseSolver(size_t nr_xs);
    ~SparseSolver();

    //! set number of threads of the solver to override global setting; set to 0
    //! to use global setting
    static void set_num_threads(int nr);

    //! get the actual number of threads used by the solver (use global setting
    //! if set_num_threads() is not valid)
    static int get_num_threads();

    //! a class for building the sparse matrix
    class SparseMatBuilder {
    protected:
        ~SparseMatBuilder() = default;

    public:
        /*!
         * \brief add a constraint to the system; must be called before
         * prepare()
         *
         * \param cidx index of the constraint (i.e., the row number of the item
         *      in the sparse matrix). All items in one row must be added before
         *      moving to the next row. The value must start from \p cidx_offset
         *      given in make_builder().
         * \param xidx index of the unknown (i.e., the col number of the item in
         *      the sparse matrix)
         * \param coeff coefficient of this unknown
         */
        virtual void add_constraint(size_t cidx, size_t xidx, fp_t coeff) = 0;
    };

    //! whether internal information should be printed
    static void set_verbosity(int verbosity);

    /*!
     * \brief make a builder for a subset of the matrix
     * \param cidx_offset offset of the \p cidx value in
     *      SparseMatBuilder::add_constraint().
     */
    SparseMatBuilder* make_builder(size_t cidx_offset = 0);

    /*!
     * \brief Prepare the solver after all constraints have been added
     *
     * The constraints are added by SparseMatBuilder::add_constraint.
     * All the SparseMatBuilder pointers are invalidated after calling
     * prepare().
     *
     * \param l2_penalty minimize \f$(Ax-b)^2 + l2_penalty*x^2\f$ rather than
     *      solve \f$Ax = b\f$
     */
    SparseSolver& prepare(fp_t l2_penalty = 0);

    //! solve the system for a given vector \p b
    TensorND solve(const TensorND& b) const;

    //! dump the equation to a file for debugging; the tensor b can be empty
    void dump(const TensorND& b, FILE* fout = stdout) const;

    //! apply the coefficient on given tensor; usually for debug
    TensorND apply(const TensorND& x) const;

    //! get the L2 norm of the coefficients
    fp_t coeff_l2() const;
};

}  // namespace sanm
