#define CF_USE_TIMER 1
#include "main.h"
#include "arap_material.h"
#include "hcheck.h"
#include "neohookean_material.h"
#include "tet_elastic_body.h"
#include "typedefs.h"
#include "utils.h"

#include <mkl.h>
#include <Eigen/PardisoSupport>

#include <cstdio>
#include <memory>

using namespace baseline;

bool baseline::g_hessian_proj = true;
double baseline::g_hessian_diag_reg = 0;
using SparseMat = materials::ElasticBody<3, double>::SparseMat;

namespace {

//! change g_hessian_proj in the scope of this class and restore to old value
class ChangeHeissianProj : public cf::NonCopyable {
    const bool m_old;

public:
    explicit ChangeHeissianProj(bool v) : m_old{g_hessian_proj} {
        g_hessian_proj = v;
    }
    ~ChangeHeissianProj() { g_hessian_proj = m_old; }
};

// compute stiffness matrix K for current configuration
void get_stiffmat_and_force(
        const materials::TetMesh<double>&
                tet_mesh,  // the tet mesh with rest configuration
        const materials::TetElasticBody<double>& tet_def_body,
        const CoordMat& vertices,  // the vertices coordinates in current
                                   // configuration (size 3 * N)
        const CoordMat* f_ext,     // static external force
        const std::vector<bool>& index_mask,  // the index mask, which obtained
                                              // from set_boundary_conditions
        SparseMat* K,                         // output: the stiffness matrix
        Eigen::VectorXd& force_elastic  // output: the elastic force in current
                                        // configuration
) {
    const size_t rows = tet_mesh.vertex().rows();
    const size_t cols = tet_mesh.vertex().cols();
    const size_t total_dim = rows * cols;

    SparseMat K_full;

    if (K) {
        K_full = tet_def_body.compute_stiffness_matrix(vertices);

        if (g_hessian_diag_reg) {
            SparseMat regularizer(K_full.rows(), K_full.cols());
            regularizer.setIdentity();

            // add a regularization term on the diagonal
            K_full += regularizer * g_hessian_diag_reg;
        }
    }

    Eigen::Matrix3Xd force_elastic_full = tet_def_body.compute_force(vertices);
    if (f_ext) {
        force_elastic_full += *f_ext;
    }

    // trim K_full and force_elastic_full with vertex_mask
    std::unique_ptr<int[]> index_mapping(new int[total_dim]);
    int dofs = 0;
    for (size_t i = 0; i < total_dim; i++) {
        index_mapping[i] = dofs;
        if (index_mask[i]) {
            dofs++;
        }
    }

    if (K) {
        std::vector<Eigen::Triplet<double>> triplet_list;
        triplet_list.reserve(K_full.nonZeros());
        for (int k = 0; k < K_full.outerSize(); k++) {
            for (SparseMat::InnerIterator it(K_full, k); it; ++it) {
                if (index_mask[it.row()] && index_mask[it.col()]) {
                    triplet_list.push_back(Eigen::Triplet<double>(
                            index_mapping[it.row()], index_mapping[it.col()],
                            it.value()));
                }
            }
        }
        K->resize(dofs, dofs);
        K->setZero();
        K->makeCompressed();
        K->setFromTriplets(triplet_list.begin(), triplet_list.end());
    }

    force_elastic = Eigen::VectorXd::Zero(dofs);
    for (size_t i = 0; i < total_dim; i++) {
        if (index_mask[i])
            force_elastic[index_mapping[i]] = force_elastic_full(i % 3, i / 3);
    }
}

std::unique_ptr<materials::Material<3, double>> make_material(
        const MaterialDesc& desc) {
    std::unique_ptr<materials::Material<3, double>> ret;
    switch (desc.energy) {
        case MaterialDesc::NEOHOOKEAN_C:
            ret.reset(new materials::CompressibleNeohookeanMaterial<3, double>(
                    desc.young, desc.poisson));
            break;
        case MaterialDesc::NEOHOOKEAN_I:
            ret.reset(
                    new materials::IncompressibleNeohookeanMaterial<3, double>(
                            desc.young, desc.poisson));
            break;
        case MaterialDesc::ARAP:
            ret.reset(new materials::ARAPElasticityMaterial<3, double>(
                    desc.young, desc.poisson));
            break;
        default:
            cf_assert(0, "bad material");
    }
    return ret;
}

void check_hessian() {
    static bool enable = getenv("FEA_CHECK");
    if (!enable) {
        return;
    }
    static bool done = false;
    if (!done) {
        ChangeHeissianProj disable_proj{false};
        materials::CompressibleNeohookeanMaterial<3, double> m0(1.2, 0.45);
        materials::check_hessian("neohookean_c", m0);
        materials::IncompressibleNeohookeanMaterial<3, double> m1(1.2, 0.45);
        materials::check_hessian("neohookean_i", m1);
        materials::ARAPElasticityMaterial<3, double> m2(1.2, 0.45);
        materials::check_hessian("arap", m2);
        done = true;
    }
}

class SparseSolver {
    bool m_self_adj;
    Eigen::PardisoLLT<SparseMat, Eigen::Upper> m_solver_llt;
    Eigen::PardisoLU<SparseMat> m_solver_lu;

public:
    explicit SparseSolver(bool self_adj) : m_self_adj{self_adj} {}

    void set_self_adj(bool self_adj) { m_self_adj = self_adj; }

    Eigen::VectorXd solve(const SparseMat& A, const Eigen::VectorXd& b) {
        Eigen::VectorXd r;
        Eigen::ComputationInfo info;
        if (m_self_adj) {
            r = m_solver_llt.compute(A).solve(b);
            info = m_solver_llt.info();
        } else {
            r = m_solver_lu.compute(A).solve(b);
            info = m_solver_lu.info();
        }
        if (info) {
            std::string info_message;
            if (info == 1) {
                info_message = "NumericalIssue";
            } else if (info == 2) {
                info_message = "NoConvergence";
            } else if (info == 3) {
                info_message = "InvalidInput";
            } else {
                info_message = "UnknownError";
            }
            throw std::runtime_error{std::move(info_message)};
        }
        return r;
    }
};

//! compute A'A with MKL builtin
void compute_aTa(SparseMat& dst, SparseMat& A) {
    cf_assert(A.isCompressed() && A.IsRowMajor);
    sparse_matrix_t A_mkl, ATA_mkl;
    {
        int* rows_start = A.outerIndexPtr();
        int* rows_end = A.outerIndexPtr() + 1;
        int* col_indx = A.innerIndexPtr();
        auto err = mkl_sparse_d_create_csr(&A_mkl, SPARSE_INDEX_BASE_ZERO,
                                           A.rows(), A.cols(), rows_start,
                                           rows_end, col_indx, A.valuePtr());
        cf_assert(err == SPARSE_STATUS_SUCCESS);
    }

    CF_DEFER(std::bind(mkl_sparse_destroy, A_mkl));

    auto status = mkl_sparse_syrk(SPARSE_OPERATION_TRANSPOSE, A_mkl, &ATA_mkl);
    cf_assert(status == SPARSE_STATUS_SUCCESS,
              "failed to compute A'A: status=%d", static_cast<int>(status));
    CF_DEFER(std::bind(mkl_sparse_destroy, ATA_mkl));

    sparse_index_base_t ATA_idx;
    MKL_INT r, c, *ATA_ia0, *ATA_ia1, *ATA_ja;
    double* ATA_a;
    status = mkl_sparse_d_export_csr(ATA_mkl, &ATA_idx, &r, &c, &ATA_ia0,
                                     &ATA_ia1, &ATA_ja, &ATA_a);
    cf_assert(status == SPARSE_STATUS_SUCCESS,
              "failed to export A'A: status=%d", static_cast<int>(status));
    cf_assert(r == c && r == A.rows());
    cf_assert(ATA_idx == SPARSE_INDEX_BASE_ZERO);
    cf_assert(ATA_ia1 == ATA_ia0 + 1);
    dst.makeCompressed();
    dst = Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>>{
            r, c, ATA_ia0[r], ATA_ia0, ATA_ja, ATA_a};
    cf_assert(dst.isCompressed());
}

double rms(const Eigen::VectorXd& v) {
    return std::sqrt(v.squaredNorm() / v.size());
}

}  // anonymous namespace

baseline::Stat baseline::solve_energy_min(
        const IndexMat& elements, const CoordMat& vtx_init,
        const CoordMat& vtx_dst, const CoordMat* f_ext, const MaskMat& bnd_mask,
        const MaterialDesc& material_desc, double gtol_refine,
        const IterCallback& iter_callback) {
    check_hessian();
    auto material = make_material(material_desc);
    materials::TetMesh<double> tet_mesh{vtx_init, elements};
    materials::TetElasticBody<double> tet_def_body(*material, {}, {}, tet_mesh);

    CoordMat vertices = vtx_dst;
    SparseMat hessian;
    Eigen::VectorXd grad;
    std::vector<bool> index_mask(bnd_mask.cols() * 3);
    for (int i = 0; i < bnd_mask.cols(); ++i) {
        for (int j = 0; j < 3; ++j) {
            index_mask[i * 3 + j] = !bnd_mask(j, i);
        }
    }

    Stat stat;
    cf::Timer timer;
    timer.start();

    constexpr double gtol = 1e-6, xtol = 1e-6, ls_c1 = 0.2;
    SparseSolver solver{g_hessian_proj};

    auto get_energy = [&tet_def_body, f_ext, &vtx_init](const CoordMat& vtx) {
        auto e = tet_def_body.compute_energy(vtx);
        if (f_ext) {
            Eigen::Map<const Eigen::VectorXd> v0_flat{vtx_init.data(),
                                                      vtx_init.size()},
                    v1_flat{vtx.data(), vtx.size()},
                    f_flat{f_ext->data(), f_ext->size()};
            e += (v0_flat - v1_flat).dot(f_flat);
        }
        return e;
    };

    Eigen::Matrix<double, 3, Eigen::Dynamic> new_vertices;
    for (;;) {
        get_stiffmat_and_force(tet_mesh, tet_def_body, vertices, f_ext,
                               index_mask, &hessian, grad);
        // this is actually the negative grad since it is the force
        ++stat.nr_iter;
        cf::Timer timer_newton;
        timer_newton.start();
        Eigen::VectorXd u = solver.solve(hessian, grad);

        double step = 1, energy = get_energy(vertices),
               c1_g_p = -ls_c1 * u.dot(grad), new_energy,
               dx_base = u.norm() / (new_vertices.norm() + 1);
        if (g_hessian_proj) {
            cf_assert(c1_g_p < 0);
        } else {
            if (c1_g_p >= 0) {
                c1_g_p = 0;
            }
        }
        int linear_search_cnt = 0;
        for (;; ++linear_search_cnt) {
            new_vertices = vertices;

            // get the final positions and shove them in u
            int cnt = 0;
            for (int i = 0; i < vertices.cols(); i++) {
                for (int j = 0; j < 3; j++) {
                    if (index_mask[i * 3 + j]) {
                        new_vertices(j, i) += u(cnt++) * step;
                    }
                }
            }

            new_energy = get_energy(new_vertices);
            if (new_energy < energy + step * c1_g_p) {
                break;
            }
            step /= 2;
            if (dx_base * step < xtol) {
                new_vertices = vertices;
                break;
            }
        }

        timer_newton.stop();
        stat.tot_newton_time += timer_newton.time();

        double grad_rms = rms(grad),
               df = (energy - new_energy) / (new_energy + 1),
               dx = dx_base * step;

        vertices = new_vertices;

        printf("\riter=%d grms=%g df=%g dx=%g step=%g f=%g cnt=%d  ",
               stat.nr_iter, grad_rms, df, dx, step, energy, linear_search_cnt);
        fflush(stdout);
        if (iter_callback && !iter_callback(vertices)) {
            break;
        }
        if (grad_rms < gtol || dx < xtol) {
            stat.df = df;
            stat.dx = dx;
            stat.grad_rms = grad_rms;
            stat.energy = energy;
            break;
        }
    }
    printf("\n");
    auto postprocess = [&]() {
        timer.stop();
        stat.tot_time = timer.time();
        stat.vtx = vertices;
    };

    if (stat.grad_rms <= gtol_refine) {
        postprocess();
        return stat;
    }

    printf("run refinement: grad_rms=%g gtol_refine=%g\n", stat.grad_rms,
           gtol_refine);

    ChangeHeissianProj disable_proj{false};
    solver.set_self_adj(false);
    get_stiffmat_and_force(tet_mesh, tet_def_body, vertices, f_ext, index_mask,
                           &hessian, grad);
    for (;;) {
        ++stat.nr_iter;
        ++stat.nr_iter_refine;
        cf::Timer timer_newton;
        timer_newton.start();
        Eigen::VectorXd u = solver.solve(hessian, grad);
        {
            int cnt = 0;
            for (int i = 0; i < vertices.cols(); i++) {
                for (int j = 0; j < 3; j++) {
                    if (index_mask[i * 3 + j]) {
                        vertices(j, i) += u(cnt++);
                    }
                }
            }
        }
        timer_newton.stop();
        stat.tot_newton_time += timer_newton.time();

        get_stiffmat_and_force(tet_mesh, tet_def_body, vertices, f_ext,
                               index_mask, &hessian, grad);

        double grad_rms = rms(grad);
        printf("\riter=%d grms=%g ", stat.nr_iter, grad_rms);
        fflush(stdout);
        if (iter_callback && !iter_callback(vertices)) {
            break;
        }
        if (grad_rms < gtol_refine || stat.nr_iter_refine >= 20) {
            stat.grad_rms_refine = grad_rms;
            break;
        }
    }
    printf("\n");
    postprocess();
    return stat;
}

baseline::Stat baseline::solve_force_equ_levmar(
        const IndexMat& elements, const CoordMat& vtx_init,
        const CoordMat& f_ext, const MaskMat& bnd_mask,
        const MaterialDesc& material_desc, double gtol,
        const IterCallback& iter_callback) {
    check_hessian();
    cf_assert(!g_hessian_proj);
    auto material = make_material(material_desc);
    materials::TetMesh<double> tet_mesh{vtx_init, elements};
    materials::TetElasticBody<double> tet_def_body(*material, {}, {}, tet_mesh);

    CoordMat vertices = vtx_init, new_vertices;
    SparseMat jacobian, damped;
    Eigen::VectorXd force, new_force, eqn_rhs;
    std::vector<bool> index_mask(bnd_mask.cols() * 3);
    for (int i = 0; i < bnd_mask.cols(); ++i) {
        for (int j = 0; j < 3; ++j) {
            index_mask[i * 3 + j] = !bnd_mask(j, i);
        }
    }

    Stat stat;
    cf::Timer timer;

    auto postprocess = [&]() {
        timer.stop();
        printf("\n");
        stat.tot_time = timer.time();
        stat.vtx = vertices;
    };

    timer.start();

    constexpr int max_iters = 1000;
    constexpr double damp_coeff_k = 10,
                     damp_coeff_min = std::numeric_limits<double>::epsilon();
    SparseSolver solver{true};
    double damp_coeff = 1e-4;
    Eigen::VectorXd delta, diag;
    for (;;) {
        ++stat.nr_iter;
        get_stiffmat_and_force(tet_mesh, tet_def_body, vertices, &f_ext,
                               index_mask, &jacobian, force);

        cf::Timer timer_newton;
        timer_newton.start();
        double energy = rms(force), new_energy;
        compute_aTa(damped, jacobian);
        eqn_rhs = jacobian.transpose() * force;
        diag = damped.diagonal();

        int search_cnt = 0;
        for (; ++search_cnt;) {
            damped.diagonal() = diag * (1 + damp_coeff);
            delta = solver.solve(damped, eqn_rhs);

            new_vertices = vertices;
            {
                // get the final positions and shove them in u
                int cnt = 0;
                for (int i = 0; i < vertices.cols(); i++) {
                    for (int j = 0; j < 3; j++) {
                        if (index_mask[i * 3 + j]) {
                            new_vertices(j, i) += delta(cnt++);
                        }
                    }
                }
            }

            timer_newton.stop();
            try {
                get_stiffmat_and_force(tet_mesh, tet_def_body, new_vertices,
                                       &f_ext, index_mask, nullptr, new_force);
                new_energy = rms(new_force);
            } catch (materials::NumericalError&) {
                new_energy = energy * 1.1;
            }
            timer_newton.start();
            if (new_energy < energy) {
                damp_coeff =
                        std::max(damp_coeff / damp_coeff_k, damp_coeff_min);
                break;
            } else {
                damp_coeff *= damp_coeff_k;
            }

            if (search_cnt >= 50) {
                // no progress can be made ...
                stat.grad_rms = energy;
                stat.dx = -1;
                postprocess();
                return stat;
            }
        }

        timer_newton.stop();
        stat.tot_newton_time += timer_newton.time();

        auto dx = delta.norm() / (new_vertices.norm() + 1);
        printf("\riter=%d grms=%g dg=%g dx=%g lambda=%g/%.1g cnt=%d  ",
               stat.nr_iter, new_energy, energy - new_energy, dx, damp_coeff,
               damp_coeff_min, search_cnt);
        fflush(stdout);

        vertices = new_vertices;
        energy = new_energy;

        if (iter_callback && !iter_callback(vertices)) {
            break;
        }
        if (energy < gtol || stat.nr_iter >= max_iters) {
            stat.dx = dx;
            stat.grad_rms = energy;
            break;
        }
    }
    postprocess();
    return stat;
}
