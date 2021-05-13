/**
 * \file libsanm/sparse_solver.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/sparse_solver.h"

#include <mkl_pardiso.h>
#include <mkl_service.h>
#include <mkl_spblas.h>
#include <mkl_types.h>

#include <cmath>
#include <cstring>

// set to 1 to construct a dense mat and print some statistics for debug
#define PRINT_MAT 0

#if PRINT_MAT
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#endif

using namespace sanm;

namespace {

class ScopedMKLThreads {
public:
    explicit ScopedMKLThreads(int nr) {
        sanm_assert(nr > 0);
        mkl_set_num_threads_local(nr);
    }

    ~ScopedMKLThreads() { mkl_set_num_threads_local(0); }
};

}  // anonymous namespace

// use Intel PARDISO. See examples_core_c.tgz/solverc/source/pardiso_unsym_c.c

class SparseSolver::Impl final : public NonCopyable {
    const size_t m_nr_unknown;
    const int m_nr_threads;
    bool m_prepared = false;

    class MatBuilder;

    std::vector<MKL_INT> m_ia, m_ja;
    std::vector<double> m_a;

    //! pointers for A'*A + p*I
    sparse_matrix_t m_aTapI = nullptr;  //!< A'A
    MKL_INT *m_aTapI_ia0 = nullptr, *m_aTapI_ia1 = nullptr,
            *m_aTapI_ja = nullptr;
    double* m_aTapI_a = nullptr;
    TensorND m_aTapI_b;
    sparse_index_base_t m_aTapI_idx{};

    fp_t m_l2_penalty = 0;

    sparse_matrix_t m_a_csr = nullptr;  //!< cached CSR descriptor
    void* m_pt[64];
    MKL_INT m_iparm[64];
    MKL_INT m_mtype = -1;
    std::vector<std::unique_ptr<MatBuilder>> m_mat_builders;

    TensorND m_solution_cache;

    void call_pardiso(MKL_INT phase, const double* b, double* x) {
        if (m_mtype == -1) {
            // initialize at first call

            if (phase == -1) {
                // destroy before first init
                return;
            }

            init_pardiso();
        }
        sanm_assert(m_prepared);

        MKL_INT maxfct = 1;  // max number of numerical factorizations
        MKL_INT mnum = 1;    // 1 <= mnum <= maxfct
        MKL_INT n = m_nr_unknown;
        MKL_INT nrhs = 1;  // number of right hand sides
        MKL_INT msglvl = sm_verbosity;
        MKL_INT error = 0;
        double* pa;
        MKL_INT *ia, *ja;
        if (m_l2_penalty == 0) {
            pa = m_a.data();
            ia = m_ia.data();
            ja = m_ja.data();
        } else {
            pa = m_aTapI_a;
            ia = m_aTapI_ia0;
            ja = m_aTapI_ja;
        }
        pardiso(m_pt, &maxfct, &mnum, &m_mtype, &phase, &n, pa, ia, ja, nullptr,
                &nrhs, m_iparm, &msglvl, const_cast<double*>(b), x, &error);
        sanm_assert(error == 0, "pardiso phase=%d failed: error=%d",
                    static_cast<int>(phase), error);
    }

    void init_pardiso() {
        if (m_l2_penalty) {
            // real and symmetric positive definite
            m_mtype = 2;
        } else {
            // real and nonsymmetric matrix
            m_mtype = 11;
        }

        pardisoinit(m_pt, &m_mtype, m_iparm);

        m_iparm[17] = 0; /* I/O: Number of nonzeros in the factor LU */
        m_iparm[18] = 0; /* I/O: Mflops for LU factorization */
        m_iparm[34] = 1; /* Zero-based indexing */

        if (m_nr_threads > 1) {
            // The parallel (OpenMP) version of the nested dissection
            // algorithm
            m_iparm[1] = 3;
        }
    }

    //! get a descriptor for the coeff matrix
    sparse_matrix_t a_csr() {
        if (!m_a_csr) {
            auto err = mkl_sparse_d_create_csr(&m_a_csr, SPARSE_INDEX_BASE_ZERO,
                                               m_nr_unknown, m_nr_unknown,
                                               m_ia.data(), m_ia.data() + 1,
                                               m_ja.data(), m_a.data());
            sanm_assert(err == SPARSE_STATUS_SUCCESS,
                        "failed to create sparse matrix: err=%d",
                        static_cast<int>(err));
        }
        return m_a_csr;
    }

public:
    static int sm_verbosity;

    Impl(size_t nr_xs) : m_nr_unknown{nr_xs}, m_nr_threads{get_num_threads()} {
        sanm_assert(m_nr_threads >= 1);
    }

    ~Impl();

    void prepare(fp_t l2_penalty);

    TensorND solve(const TensorND& b) {
        SANM_SCOPED_PROFILER("sparse_solve");
        ScopedMKLThreads mkl_threads{m_nr_threads};
        sanm_assert(b.shape().total_nr_elems() == m_nr_unknown);
        auto bptr = b.ptr();
        for (size_t i = 0; i < m_nr_unknown; ++i) {
            sanm_assert(std::isfinite(bptr[i]), "b[%zu]=%g", i, bptr[i]);
        }
        if (m_l2_penalty) {
            m_aTapI_b.set_shape({m_nr_unknown});
            matrix_descr desc{
                    .type = SPARSE_MATRIX_TYPE_GENERAL,
                    .mode = SPARSE_FILL_MODE_UPPER,
                    .diag = SPARSE_DIAG_NON_UNIT,
            };
            auto status =
                    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1, a_csr(),
                                    desc, bptr, 0, m_aTapI_b.woptr());
            sanm_assert(status == SPARSE_STATUS_SUCCESS,
                        "failed to compute A'b: status=%d",
                        static_cast<int>(status));
            bptr = m_aTapI_b.ptr();
        }
        call_pardiso(33, bptr,
                     m_solution_cache.set_shape({m_nr_unknown}).woptr());
        return m_solution_cache;
    }

    void dump(const TensorND& b, FILE* fout) {
        sanm_assert(m_prepared);
        sanm_assert(b.empty() || (b.shape().total_nr_elems() == m_nr_unknown));
        auto bptr = b.empty() ? nullptr : b.ptr();
        fprintf(fout, "======== begin SparseSolver_%p dump ========\n", this);
        for (size_t i = 0; i < m_nr_unknown; ++i) {
            fprintf(fout, "<eqn%-3zu>: ", i);
            for (int j = m_ia[i]; j < m_ia[i + 1]; ++j) {
                fprintf(fout, "%3.3fx%-5zu", m_a[j],
                        static_cast<size_t>(m_ja[j]));
            }
            if (bptr) {
                fprintf(fout, " = %g\n", bptr[i]);
            } else {
                fprintf(fout, "\n");
            }
        }
        fprintf(fout, "======== end SparseSolver_%p dump ========\n", this);
    }

    TensorND apply(const TensorND& x) {
        ScopedMKLThreads mkl_threads{m_nr_threads};
        sanm_assert(m_prepared && x.rank() == 1 && x.shape(0) == m_nr_unknown);
        auto ret = x.make_same_shape();
        matrix_descr desc{.type = SPARSE_MATRIX_TYPE_GENERAL,
                          .mode = SPARSE_FILL_MODE_UPPER,
                          .diag = SPARSE_DIAG_NON_UNIT};
        auto err = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, a_csr(),
                                   desc, x.ptr(), 0, ret.woptr());
        sanm_assert(err == SPARSE_STATUS_SUCCESS,
                    "failed to compute sparse mv: err=%d",
                    static_cast<int>(err));
        return ret;
    }

    fp_t coeff_l2() const {
        fp_t s = 0;
        for (fp_t i : m_a) {
            s += i * i;
        }
        return std::sqrt(s);
    }

    SparseMatBuilder* make_builder(size_t cidx_offset);
};
int SparseSolver::Impl::sm_verbosity = 0;

class SparseSolver::Impl::MatBuilder final
        : public SparseSolver::SparseMatBuilder {
    struct CsrRowPair {
        leastsize_t col;
        double val;

        CsrRowPair() = default;
        CsrRowPair(size_t c, double v)
                : col{static_cast<leastsize_t>(c)}, val{v} {}

        bool operator<(const CsrRowPair& rhs) const { return col < rhs.col; }
    };

    Impl* const m_owner;
    const size_t m_offset;
    size_t m_prev_constraint_cidx;
    std::vector<CsrRowPair> m_csr_last_row;

    std::vector<MKL_INT> m_ia, m_ja;
    std::vector<double> m_a;

    void flush_csr_row() {
        sanm_assert(!m_csr_last_row.empty(), "empty row %zu",
                    m_prev_constraint_cidx);
        std::sort(m_csr_last_row.begin(), m_csr_last_row.end());
        size_t jr = 0, jsize = m_csr_last_row.size();
        m_csr_last_row.emplace_back(m_owner->m_nr_unknown,
                                    0.);  // sentinel value

        if (size_t tsize = m_a.size() + m_csr_last_row.size();
            m_a.capacity() < tsize) {
            tsize = std::max(tsize, m_a.capacity() * 3 / 2);
            m_a.reserve(tsize);
            m_ja.reserve(tsize);
        }

        m_ia.push_back(m_ja.size());
        while (jr < jsize) {
            leastsize_t c = m_csr_last_row[jr].col;
            double v = m_csr_last_row[jr].val;
            ++jr;
            while (m_csr_last_row[jr].col == c) {
                v += m_csr_last_row[jr].val;
                ++jr;
            }
            m_ja.push_back(c);
            m_a.push_back(v);
        }
        m_csr_last_row.clear();
    }

public:
    MatBuilder(Impl* owner, size_t offset)
            : m_owner{owner},
              m_offset{offset},
              m_prev_constraint_cidx{offset} {}

    void add_constraint(size_t cidx, size_t xidx, double coeff) override {
        sanm_assert(!m_owner->m_prepared);
        sanm_assert(std::isfinite(coeff), "coeff[%zu,%zu]=%g", cidx, xidx,
                    coeff);

        if (std::fabs(coeff) < 1e-9) {
            return;
        }

        sanm_assert(cidx < m_owner->m_nr_unknown &&
                    xidx < m_owner->m_nr_unknown);
        if (cidx != m_prev_constraint_cidx) {
            sanm_assert(cidx == m_prev_constraint_cidx + 1,
                        "constraints not added in order: prev=%zu cur=%zu",
                        m_prev_constraint_cidx, static_cast<size_t>(cidx));
            flush_csr_row();
            m_prev_constraint_cidx = cidx;
        }
        m_csr_last_row.emplace_back(xidx, coeff);
    }

    void prepare() {
        flush_csr_row();
        ++m_prev_constraint_cidx;
    }

    size_t ja_size() const { return m_ja.size(); }

    size_t first_cidx() const { return m_offset; }

    size_t last_cidx() const { return m_prev_constraint_cidx; }

    void copy_into(size_t ia_offset, MKL_INT* ia, MKL_INT* ja, double* a) {
        for (size_t i = m_offset; i < m_prev_constraint_cidx; ++i) {
            ia[i] = m_ia[i - m_offset] + ia_offset;
        }
        memcpy(ja + ia_offset, m_ja.data(), sizeof(MKL_INT) * m_ja.size());
        memcpy(a + ia_offset, m_a.data(), sizeof(double) * m_a.size());
    }
};

void SparseSolver::Impl::prepare(fp_t l2_penalty) {
    SANM_SCOPED_PROFILER("sparse_prep");
    sanm_assert(!m_prepared);
    m_prepared = true;
    m_l2_penalty = l2_penalty;

    ScopedMKLThreads mkl_threads{m_nr_threads};

    size_t tot_ja_size = 0;
    for (auto& i : m_mat_builders) {
        i->prepare();
        tot_ja_size += i->ja_size();
    }

    std::sort(m_mat_builders.begin(), m_mat_builders.end(),
              [](const std::unique_ptr<MatBuilder>& a,
                 const std::unique_ptr<MatBuilder>& b) {
                  return a->first_cidx() < b->first_cidx();
              });

    for (size_t i = 1; i < m_mat_builders.size(); ++i) {
        sanm_assert(m_mat_builders[i]->first_cidx() ==
                    m_mat_builders[i - 1]->last_cidx());
    }
    sanm_assert(m_mat_builders.back()->last_cidx() == m_nr_unknown);

    m_ia.resize(m_nr_unknown + 1);
    m_ja.resize(tot_ja_size);
    m_a.resize(tot_ja_size);
    m_ia.back() = tot_ja_size;
    {
        size_t offset = 0;
        for (auto& i : m_mat_builders) {
            i->copy_into(offset, m_ia.data(), m_ja.data(), m_a.data());
            offset += i->ja_size();
        }
    }
    m_mat_builders.clear();

    if (m_l2_penalty) {
        SANM_SCOPED_PROFILER("sparse_A'A");
        auto status =
                mkl_sparse_syrk(SPARSE_OPERATION_TRANSPOSE, a_csr(), &m_aTapI);
        sanm_assert(status == SPARSE_STATUS_SUCCESS,
                    "failed to compute A'A: status=%d",
                    static_cast<int>(status));
        MKL_INT r, c;
        status = mkl_sparse_d_export_csr(m_aTapI, &m_aTapI_idx, &r, &c,
                                         &m_aTapI_ia0, &m_aTapI_ia1,
                                         &m_aTapI_ja, &m_aTapI_a);
        sanm_assert(status == SPARSE_STATUS_SUCCESS,
                    "failed to export A'A: status=%d",
                    static_cast<int>(status));

        sanm_assert(r == c && static_cast<size_t>(r) == m_nr_unknown);

        sanm_assert(m_aTapI_idx == SPARSE_INDEX_BASE_ZERO);
        sanm_assert(m_aTapI_ia1 == m_aTapI_ia0 + 1);
        for (int i = 0; i < r; ++i) {
            int p = m_aTapI_ia0[i], c = m_aTapI_ja[p];
            sanm_assert(i == c, "row %d: first col is %d", i, c);
            m_aTapI_a[p] += m_l2_penalty;
        }
        if (sm_verbosity) {
            printf("Linear solve with L2: n=%zu size_A=%zu size_A'A=%zu\n",
                   m_nr_unknown, static_cast<size_t>(m_ia[m_nr_unknown]),
                   static_cast<size_t>(m_aTapI_ia1[m_nr_unknown - 1]));
        }
    }

#if PRINT_MAT
    {
        using Mat = Eigen::Matrix<fp_t, Eigen::Dynamic, Eigen::Dynamic>;
        Mat M(m_nr_unknown, m_nr_unknown);
        M.setZero();
        for (size_t i = 0; i < m_nr_unknown; ++i) {
            for (int j = m_ia[i]; j < m_ia[i + 1]; ++j) {
                M(i, m_ja[j]) = m_a[j];
            }
        }
        printf("======== dump sparse solver coeff: %zu\n", m_nr_unknown);
        if (m_nr_unknown <= 10) {
            Eigen::IOFormat fmt(3, 0, ", ", ";\n", "", "", "[", "]");
            std::cout << M.format(fmt) << std::endl;
        }
        Eigen::BDCSVD<Mat> svd(M);
        std::cout << "s: " << svd.singularValues().transpose() << std::endl;
        std::cout << "det=" << M.determinant() << std::endl;
        std::cout << "======================\n";
    }
#endif  // PRINT_MAT

    // analysis and factorization
    call_pardiso(12, nullptr, nullptr);
}

SparseSolver::SparseMatBuilder* SparseSolver::Impl::make_builder(
        size_t cidx_offset = 0) {
    auto& ptr = m_mat_builders.emplace_back();
    ptr.reset(new MatBuilder{this, cidx_offset});
    return ptr.get();
}

SparseSolver::Impl::~Impl() {
    call_pardiso(-1, nullptr, nullptr);
    if (m_a_csr) {
        mkl_sparse_destroy(m_a_csr);
        m_a_csr = nullptr;
    }
    if (m_aTapI) {
        mkl_sparse_destroy(m_aTapI);
        m_aTapI = nullptr;
    }
}

void SparseSolver::set_verbosity(int verbosity) {
    Impl::sm_verbosity = verbosity;
}

SparseSolver::SparseSolver(size_t nr_xs) {
    m_pimpl.reset(new Impl{nr_xs});
}

SparseSolver::~SparseSolver() = default;

SparseSolver::SparseMatBuilder* SparseSolver::make_builder(size_t cidx_offset) {
    return m_pimpl->make_builder(cidx_offset);
}

SparseSolver& SparseSolver::prepare(fp_t l2_penalty) {
    m_pimpl->prepare(l2_penalty);
    return *this;
}

TensorND SparseSolver::solve(const TensorND& b) const {
    return m_pimpl->solve(b);
}

void SparseSolver::dump(const TensorND& b, FILE* fout) const {
    m_pimpl->dump(b, fout);
}

TensorND SparseSolver::apply(const TensorND& x) const {
    return m_pimpl->apply(x);
}

fp_t SparseSolver::coeff_l2() const {
    return m_pimpl->coeff_l2();
}

static int g_solver_num_threads = 0;
void SparseSolver::set_num_threads(int nr) {
    sanm_assert(nr >= 0);
    g_solver_num_threads = nr;
}

int SparseSolver::get_num_threads() {
    if (g_solver_num_threads) {
        return g_solver_num_threads;
    }
    return sanm::get_num_threads();
}
