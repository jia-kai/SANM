/**
 * \file fea/main.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "fea/baseline/main.h"
#include "fea/tetrahedral_mesh.h"
#include "libsanm/anm.h"
#include "libsanm/sparse_solver.h"

#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include <sys/resource.h>
#include <sys/time.h>

using namespace fea;
using namespace sanm;

using json = nlohmann::json;

namespace {

constexpr double RMS_THRESH_FORCE_EQU = 1e-10;
int g_total_nr_iter = 0;

const std::unordered_map<std::string, EnergyModel> NAME2ENERGY_MODEL{
        {"neohookean_c", EnergyModel::NEOHOOKEAN_C},
        {"neohookean_i", EnergyModel::NEOHOOKEAN_I},
        {"arap", EnergyModel::ARAP},
};
const std::unordered_map<std::string, baseline::MaterialDesc::Energy>
        NAME2BASELINE_ENERGY_MODEL{
                {"neohookean_c", baseline::MaterialDesc::NEOHOOKEAN_C},
                {"neohookean_i", baseline::MaterialDesc::NEOHOOKEAN_I},
                {"arap", baseline::MaterialDesc::ARAP},
        };

void save_mesh(const std::string& filename, const TetrahedralMesh& mesh,
               const std::unordered_set<int>* surface = nullptr) {
    FILE* fout = fopen(filename.c_str(), "w");
    sanm_assert(fout);
    SANM_DEFER(std::bind(::fclose, fout));
    mesh.write_to_file(fout, surface);
}

void save_json(const std::string& filename, const json& j) {
    FILE* fout = fopen(filename.c_str(), "w");
    sanm_assert(fout);
    SANM_DEFER(std::bind(::fclose, fout));
    fprintf(fout, "%s\n", j.dump(4).c_str());
}

Vec3 json_get_vec3(const json& j) {
    sanm_assert(j.is_array() && j.size() == 3);
    Vec3 ret;
    for (int i = 0; i < 3; ++i) {
        ret[i] = j[i].get<double>();
    }
    return ret;
}

void save_out_surface_vtx(const json& config, const TetrahedralMesh& mesh) {
    if (config.contains("out_surface_vtx")) {
        FILE* fout = fopen(config["out_surface_vtx"].get<std::string>().c_str(),
                           "w");
        sanm_assert(fout);
        mesh.write_to_surface_vtx_file(fout);
        SANM_DEFER(std::bind(::fclose, fout));
    }
}

bool json_get_bool_opt(const json& j, const char* key, bool default_ = false) {
    if (!j.contains(key)) {
        return default_;
    }
    return j[key].get<bool>();
}

template <typename T>
T json_get_opt(const json& j, const char* key, T default_) {
    if (!j.contains(key)) {
        return default_;
    }
    return j[key].get<T>();
}

void setup_solver_param(const json& config,
                        ANMDriverHelper::HyperParam& param) {
    param.order = json_get_opt(config, "order", 20);
    param.xcoeff_l2_penalty =
            json_get_opt<double>(config, "xcoeff_l2_penalty", 0);
    param.use_pade = !json_get_bool_opt(config, "disable_pade", false);
    param.sanity_check =
            !json_get_bool_opt(config, "disable_anm_sanity_check", false);
}

void setup_solver_param(const json& config, ANMEqnSolver::HyperParam& param) {
    setup_solver_param(config,
                       static_cast<ANMDriverHelper::HyperParam&>(param));
    param.converge_rms = RMS_THRESH_FORCE_EQU;
}

//! try to setup global params in baseline solver; return true if we should run
//! baseline
bool setup_baseline(const json& config) {
    if (!config.contains("baseline")) {
        return false;
    }
    json bc = config["baseline"];
    baseline::g_hessian_proj = !json_get_bool_opt(bc, "hessian_no_proj", false);
    baseline::g_hessian_diag_reg = json_get_opt<double>(bc, "hessian_diag", 0);
    printf(": using baseline: proj=%d reg=%g\n", baseline::g_hessian_proj,
           baseline::g_hessian_diag_reg);
    return true;
}

MaterialProperty make_material_property(const json& config,
                                        bool need_density = false) {
    auto type = config["type"].get<std::string>();
    MaterialProperty ret;
    if (type == "young_poisson") {
        ret = MaterialProperty::from_young_poisson(
                config["young"].get<double>(), config["poisson"].get<double>());
    } else {
        throw std::runtime_error{
                ssprintf("unknown material type %s", type.c_str())};
    }
    if (need_density || config.contains("density")) {
        ret.set_density(config["density"].get<double>());
    }
    return ret;
}

baseline::MaterialDesc make_baseline_material_desc(const json& config) {
    auto m = make_material_property(config["material"]);
    return {NAME2BASELINE_ENERGY_MODEL.at(
                    config["energy_model"].get<std::string>()),
            m.young_modulus(), m.poisson_ratio()};
}

void make_baseline_stat(json& jstat, const baseline::Stat& stat) {
    printf("baseline: iter=%d tot_time=%g newton_time=%g\n", stat.nr_iter,
           stat.tot_time, stat.tot_newton_time);
    jstat["iter_tot"] = stat.nr_iter;
    jstat["iter_refine"] = stat.nr_iter_refine;
    jstat["df"] = stat.df;
    jstat["dx"] = stat.dx;
    jstat["force_rms"] = stat.grad_rms;
    jstat["force_rms_refine"] = stat.grad_rms_refine;
    jstat["potential"] = stat.energy;
    jstat["time"] = stat.tot_time;
    jstat["newton_time"] = stat.tot_newton_time;
}

TensorND run_anm(ANMEqnSolver& solver) {
    int iter = 0;
    for (;;) {
        printf(" %.2g", solver.residual_rms());
        if (iter && iter % 10 == 0) {
            printf("(%d)", iter);
        }
        fflush(stdout);
        if (solver.converged()) {
            break;
        }
        solver.next_iter();
        ++iter;
    }
    iter = solver.get_nr_ieter();
    g_total_nr_iter += iter;
    printf(" iter=%d\n", iter);
    return solver.get_x();
}

using ImplicitIterCB = std::function<void(const ANMImplicitSolver&)>;
TensorND run_anm(ANMImplicitSolver& solver, fp_t t_dest = 1,
                 ImplicitIterCB callback = {}) {
    int iter = 0;
    for (;;) {
        printf(" %.2g", solver.get_t_upper());
        if (iter && iter % 10 == 0) {
            printf("(%d)", iter);
        }
        fflush(stdout);
        if (callback) {
            callback(solver);
        }
        if (solver.get_t_upper() >= t_dest) {
            break;
        }
        solver.update_approx();
        ++iter;
    }
    iter = solver.get_nr_ieter();
    g_total_nr_iter += iter;
    printf(" iter=%d\n", iter);
    return solver.eval(solver.solve_a(t_dest)).first;
}

//! evaluate relative displacement of vertices to measure the overall
//! deformation
fp_t relative_displacement(const CoordMat3D& v0, const CoordMat3D& v1) {
    Vec3 vmin = v0.rowwise().minCoeff(), vmax = v0.rowwise().maxCoeff();
    fp_t d = std::sqrt((v1 - v0).squaredNorm() / v0.size());
    return d / (vmax - vmin).norm();
}

//! get number of inverted tets
int get_nr_inverted(const TetIndexMat& elements, const CoordMat3D& v0,
                    const CoordMat3D& v1) {
    auto detsign = [](const Eigen::Vector4i& ele, const CoordMat3D& v) {
        Vec3 v0 = v.col(ele[0]), x0 = v.col(ele[1]) - v0,
             x1 = v.col(ele[2]) - v0, x2 = v.col(ele[3]) - v0;
        return x0.cross(x1).dot(x2) >= 0;
    };
    int ret = 0;
    for (int i = 0; i < elements.cols(); ++i) {
        auto&& e = elements.col(i);
        if (detsign(e, v0) != detsign(e, v1)) {
            ++ret;
            if (ret <= 10) {
                printf("****** WARNING: tet %d is inverted\n", i);
            }
        }
    }
    return ret;
}

//! solve static equilibrium problem
TetrahedralMeshPtr run_and_save(const char* name, const json& config,
                                const TetrahedralDeformableBody& deformable,
                                bool inverse_mode,
                                const CoordMat3D& f_load_full, bool save = true,
                                bool allow_invcheck = true) {
    printf("solving %s%s ", name, inverse_mode ? " (inv)" : "");
    fflush(stdout);
    json jstat;

    Timer timer;
    timer.start();

    auto energy_model =
            NAME2ENERGY_MODEL.at(config["energy_model"].get<std::string>());
    auto model = inverse_mode ? deformable.make_inverse(energy_model)
                              : deformable.make_forward(energy_model);
    auto f_load_sub = model->lt_inp->copy_vtx_values(f_load_full);

    auto make_out_mesh = [&](const TensorND& xt) {
        auto out_mesh = std::make_shared<TetrahedralMesh>(deformable.mesh());
        out_mesh->replace_with_mask(deformable.coord_fixed_mask(), xt);
        return out_mesh;
    };

    const double time_prep = timer.stop().time();
    jstat["time_prep"] = time_prep;

    auto post_process = [&](const TensorND& xt) {
        auto out_mesh = make_out_mesh(xt);

        auto frms = TetrahedralDeformableBody::solution_sanity_check(
                *model, xt, f_load_sub, *out_mesh);
        jstat["force_rms_recomp"] = frms;
        jstat["mesh_V"] = deformable.mesh().nr_vertices();
        jstat["mesh_F"] = deformable.mesh().nr_faces();
        jstat["displacement"] = relative_displacement(
                deformable.mesh().vertices(), out_mesh->vertices());
        jstat["nr_inverted"] = get_nr_inverted(deformable.mesh().faces(),
                                               deformable.mesh().vertices(),
                                               out_mesh->vertices());
        if (save) {
            auto out_filename = config["out_filename"].get<std::string>();
            save_mesh(out_filename + "-orig.obj", deformable.mesh());
            out_filename += ssprintf("-i%d-", inverse_mode);
            out_filename += config["energy_model"].get<std::string>();
            save_mesh(out_filename + ".obj", *out_mesh);
            save_json(out_filename + ".json", jstat);
            save_out_surface_vtx(config, *out_mesh);
        }

        if (allow_invcheck && getenv("FEA_INVCHECK")) {
            TetrahedralDeformableBody deformable_inv{deformable.material(),
                                                     out_mesh};
            deformable_inv.coord_fixed_mask() = deformable.coord_fixed_mask();
            auto restored = run_and_save(
                    (std::string(name) + " invcheck").c_str(), config,
                    deformable_inv, !inverse_mode, f_load_full, false, false);
            fp_t norm = (restored->vertices() - deformable.mesh().vertices())
                                .norm();
            printf("invcheck norm: %g\n", norm);
        }
        return out_mesh;
    };

    if (setup_baseline(config)) {
        sanm_assert(!inverse_mode);
        baseline::Stat stat;
        if (json_get_bool_opt(config["baseline"], "use_levmar")) {
            printf("opt: levmar\n");
            baseline::g_hessian_proj = false;
            stat = baseline::solve_force_equ_levmar(
                    deformable.mesh().faces(), deformable.mesh().vertices(),
                    f_load_full, deformable.coord_fixed_mask(),
                    make_baseline_material_desc(config), RMS_THRESH_FORCE_EQU);
        } else {
            stat = baseline::solve_energy_min(
                    deformable.mesh().faces(), deformable.mesh().vertices(),
                    deformable.mesh().vertices(), &f_load_full,
                    deformable.coord_fixed_mask(),
                    make_baseline_material_desc(config), RMS_THRESH_FORCE_EQU);
        }
        make_baseline_stat(jstat, stat);
        auto xt = model->lt_inp->copy_vtx_values(stat.vtx);
        return post_process(xt);
    }

    int total_nr_iter_begin = g_total_nr_iter;
    timer.reset().start();
    ANMEqnSolver::HyperParam hyper_param;
    setup_solver_param(config, hyper_param);
    hyper_param.solution_check_tol = 1e-3;

    TensorND xt;
    if (json_get_bool_opt(config, "save_interm")) {
        // larger tol because we have no error correction here
        hyper_param.solution_check_tol = 0.01;
        ANMSolverVecScale solver{model->y.node(),
                                 model->lt_inp,
                                 model->lt_out,
                                 model->lt_inp->x0(),
                                 0,
                                 f_load_sub,
                                 hyper_param};
        printf("interm: ");
        fp_t tnext = 0.1;
        for (; tnext < 1;) {
            while (tnext <= 1.05 && solver.get_t_upper() >= tnext) {
                xt = solver.eval(solver.solve_a(tnext)).first;
                save_mesh(ssprintf("%s-%.1f.obj",
                                   config["out_filename"]
                                           .get<std::string>()
                                           .c_str(),
                                   tnext),
                          *make_out_mesh(xt));
                printf(" %g", tnext);
                fflush(stdout);
                tnext += 0.1;
            }
            if (tnext >= 1) {
                break;
            }
            solver.update_approx();
        }
    } else {
        printf("order=%d:", hyper_param.order);
        fflush(stdout);
        ANMEqnSolver solver{model->y.node(),     model->lt_inp, model->lt_out,
                            model->lt_inp->x0(), f_load_sub,    hyper_param};
        xt = run_anm(solver);
        printf("timing(sec): prep=%.3f solve=%.3f\n", time_prep,
               timer.stop().time());
    }

    jstat["time_solve"] = timer.time();
    jstat["order"] = hyper_param.order;
    jstat["name"] = name;
    jstat["threads"] = get_num_threads();
    jstat["solver_threads"] = SparseSolver::get_num_threads();
    jstat["pade"] = hyper_param.use_pade;
    jstat["iter"] = g_total_nr_iter - total_nr_iter_begin;
    return post_process(xt);
}

//! note that \p vtx_coord is the init coord and will be updated inplace
json run_with_vtx_delta(const char* name, const json& config,
                        const TetrahedralDeformableBody& deformable,
                        const CoordMat3D& vtx_delta, CoordMat3D& vtx_coord,
                        bool require_refine,
                        const CoordMat3D* refine_f_load = nullptr) {
    printf("solving %s(delta) ", name);
    fflush(stdout);

    json jstat;

    Timer timer;
    timer.start();
    // vertex with delta applied
    CoordMat3D vtx_dst_boundary = deformable.mesh().vertices() + vtx_delta;

    // ensure that boundary conditions are set correctly
    auto enforce_dst_boundary = [&vtx_dst_boundary, &vtx_coord, &deformable]() {
        auto&& mask = deformable.coord_fixed_mask();
        for (int j = 0; j < mask.rows(); ++j) {
            for (int i = 0; i < 3; ++i) {
                if (mask(i, j)) {
                    vtx_coord(i, j) = vtx_dst_boundary(i, j);
                }
            }
        }
    };

    auto energy_model =
            NAME2ENERGY_MODEL.at(config["energy_model"].get<std::string>());
    auto model = deformable.make_forward(energy_model, &vtx_coord, &vtx_delta);

    auto eval_force_rms = [&]() {
        auto model = deformable.make_forward(energy_model, &vtx_coord);
        auto force = model->lt_out->apply(symbolic::eval_unary_func(
                model->y.node(), model->lt_inp->apply(model->lt_inp->x0())));
        return force.norm_rms();
    };
    auto eval_potential = [&]() -> fp_t {
        auto model = deformable.make_forward(energy_model, &vtx_coord);
        if (!model->potential.node()) {
            return -1;
        }
        auto energy = symbolic::eval_unary_func(
                model->potential.node(),
                model->lt_inp->apply(model->lt_inp->x0()));
        return energy.flatten_as_vec().reduce_sum(0).ptr()[0];
    };

    auto postprocess = [&]() {
        enforce_dst_boundary();
        jstat["force_rms_recomp"] = eval_force_rms();
        jstat["potential_recomp"] = eval_potential();
        auto& m = deformable.mesh();
        jstat["displacement"] = relative_displacement(m.vertices(), vtx_coord);
        jstat["nr_inverted"] =
                get_nr_inverted(m.faces(), m.vertices(), vtx_coord);
        jstat["V"] = m.nr_vertices();
        jstat["F"] = m.nr_faces();
    };

    int total_nr_iter_begin = g_total_nr_iter;

    if (setup_baseline(config)) {
        auto stat = baseline::solve_energy_min(
                deformable.mesh().faces(), deformable.mesh().vertices(),
                vtx_dst_boundary, nullptr, deformable.coord_fixed_mask(),
                make_baseline_material_desc(config), RMS_THRESH_FORCE_EQU);
        vtx_coord = stat.vtx;
        postprocess();
        make_baseline_stat(jstat, stat);
        return jstat;
    }

    double time_prep = timer.stop().time();
    timer.reset().start();
    ANMImplicitSolver::HyperParam hyper_param;
    setup_solver_param(config, hyper_param);
    hyper_param.solution_check_tol = 10;  // a high tol
    printf("order=%d:", hyper_param.order);
    fflush(stdout);
    ANMImplicitSolver solver{model->y.node(),
                             model->lt_inp,
                             model->lt_out,
                             model->lt_inp->x0(),
                             0,
                             hyper_param};
    ImplicitIterCB solver_cb;
    bool save_interm = json_get_bool_opt(config, "save_interm");
    if (save_interm) {
        solver_cb = [&,
                     tnext = 0._fp](const ANMImplicitSolver& solver) mutable {
            while (tnext <= 1 && solver.get_t_upper() >= tnext) {
                TensorND xt = solver.eval(solver.solve_a(tnext)).first;
                TetrahedralMesh new_mesh{vtx_coord, deformable.mesh().faces()};
                new_mesh.replace_with_mask(deformable.coord_fixed_mask(), xt);
                new_mesh.apply_vtx_delta(vtx_delta * tnext);
                save_mesh(ssprintf("%s-%.2f.obj",
                                   config["out_filename"]
                                           .get<std::string>()
                                           .c_str(),
                                   tnext),
                          new_mesh);
                tnext += 0.1;
            }
        };
    }
    auto xt = run_anm(solver, 1, solver_cb);
    timer.stop();
    printf("timing(sec): prep=%.3f solve=%.3f\n", time_prep, timer.time());
    replace_with_mask(vtx_coord, deformable.coord_fixed_mask(), xt.ptr(),
                      xt.shape().total_nr_elems());
    vtx_coord += vtx_delta;
    {
        auto force_rms = eval_force_rms();
        printf("force rms: %g\n", force_rms);
        require_refine = require_refine || force_rms >= RMS_THRESH_FORCE_EQU;
    }
    int total_nr_iter_before_refine = g_total_nr_iter;
    if (require_refine) {
        auto model = deformable.make_forward(energy_model, &vtx_coord);
        TensorND f_load_sub;
        if (refine_f_load) {
            f_load_sub = model->lt_inp->copy_vtx_values(*refine_f_load);
        } else {
            f_load_sub = model->lt_inp->x0().fill_with(0);
        }

        ANMEqnSolver::HyperParam hyper_param;
        setup_solver_param(config, hyper_param);
        hyper_param.order = 6;  // use a lower order for refinement
        timer.start();
        ANMEqnSolver solver{model->y.node(),     model->lt_inp, model->lt_out,
                            model->lt_inp->x0(), f_load_sub,    hyper_param};
        printf("refine %s:", name);
        auto xt = run_anm(solver);
        timer.stop();
        replace_with_mask(vtx_coord, deformable.coord_fixed_mask(), xt.ptr(),
                          xt.shape().total_nr_elems());
    }
    postprocess();
    jstat["iter_tot"] = g_total_nr_iter - total_nr_iter_begin;
    jstat["iter_deform"] = total_nr_iter_before_refine - total_nr_iter_begin;
    jstat["iter_refine"] = g_total_nr_iter - total_nr_iter_before_refine;
    jstat["time"] = timer.time();
    jstat["pade"] = hyper_param.use_pade;
    return jstat;
}

void test_single_tet_inverse(const json& config) {
    fp_t spacing = config["spacing"].get<double>();
    auto material = make_material_property(config["material"]);
    CoordMat3D orig_coord;
    TetIndexMat orig_tet;
    {
        fp_t angle = M_PI * 2 / 3;
        orig_coord = CoordMat3D::Zero(3, 4);
        for (int i = 0; i < 3; ++i) {
            orig_coord(0, i) = std::cos(angle * i) * spacing;
            orig_coord(1, i) = std::sin(angle * i) * spacing;
        }
        orig_coord(2, 3) = spacing;
        orig_tet.resize(4, 1);
        for (int i = 0; i < 4; ++i) {
            orig_tet(i, 0) = i;
        }
    }
    TetrahedralDeformableBody deformable{
            material, std::make_shared<TetrahedralMesh>(orig_coord, orig_tet)};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            deformable.coord_fixed_mask()(j, i) = true;
        }
    }

    CoordMat3D f_load_full =
            CoordMat3D::Zero(3, deformable.mesh().nr_vertices());
    f_load_full(2, 3) = -1000;
    auto out_mesh = run_and_save("single tet inv", config, deformable, true,
                                 f_load_full);

    for (int i = 0; i < 4; ++i) {
        Vec3 a = orig_coord.col(i), b = out_mesh->vertices().col(i);
        printf("vertex %d: (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f)\n", i, a[0],
               a[1], a[2], b[0], b[1], b[2]);
    }
}

void test_cuboid(const json& config) {
    // number of nodes (vertices) in each dimension of our mesh
    const size_t num_x_vertices = config["x"].get<int>(),
                 num_y_vertices = config["y"].get<int>(),
                 num_z_vertices = config["z"].get<int>();
    const fp_t spacing = config["spacing"].get<fp_t>();

    auto material = make_material_property(config["material"]);
    TetrahedralDeformableBody deformable{
            material,
            TetrahedralMesh::make_cuboid(num_x_vertices, num_y_vertices,
                                         num_z_vertices, spacing)};
    CoordMat3D f_load_full =
            CoordMat3D::Zero(3, deformable.mesh().nr_vertices());

    {
        // setup boundary condition
        const CoordMat3D& vtx = deformable.mesh().vertices();
        // fix the vertices on the left face and remove them from variable list
        for (int i = 0; i < vtx.cols(); ++i) {
            if (vtx(0, i) <= spacing / 2.0) {
                for (int j = 0; j < 3; j++) {
                    deformable.coord_fixed_mask()(j, i) = true;
                }
            }
        }

        // apply external forces on each nodes of bottom right region
        for (int i = 0; i < f_load_full.cols(); ++i) {
            if (vtx(0, i) >
                        (num_x_vertices / 2 - 1) * spacing - spacing / 2.0 &&
                vtx(2, i) < spacing / 2.0) {
                f_load_full(2, i) = -50.0;
            }
        }
    }

    bool inverse = json_get_bool_opt(config, "inverse");
    run_and_save(inverse ? "cuboid inverse" : "cuboid", config, deformable,
                 inverse, f_load_full);
}

void test_cuboid_twist(const json& config) {
    // number of nodes (vertices) in each dimension of our mesh
    const size_t num_x_vertices = config["x"].get<int>(),
                 num_y_vertices = config["y"].get<int>(),
                 num_z_vertices = config["z"].get<int>();
    const fp_t spacing = config["spacing"].get<fp_t>();

    auto material = make_material_property(config["material"]);
    auto mesh = TetrahedralMesh::make_cuboid(num_x_vertices, num_y_vertices,
                                             num_z_vertices, spacing);
    printf("cuboid twist: V=%zu F=%zu\n", mesh->nr_vertices(),
           mesh->nr_faces());
    TetrahedralDeformableBody deformable{material, mesh};
    fp_t x_thresh = spacing * (num_x_vertices - 1.5);
    // setup boundary condition
    CoordMat3D vtx_cur = deformable.mesh().vertices();
    std::vector<size_t> vtx_bnd_idx;
    // fix the vertices on the left face and move the right face
    for (int i = 0; i < vtx_cur.cols(); ++i) {
        if (vtx_cur(0, i) <= spacing / 2.0 || vtx_cur(0, i) >= x_thresh) {
            for (int j = 0; j < 3; j++) {
                deformable.coord_fixed_mask()(j, i) = true;
            }
            if (vtx_cur(0, i) >= x_thresh) {
                vtx_bnd_idx.push_back(i);
            }
        }
    }
    sanm_assert(!vtx_bnd_idx.empty());

    CoordMat3D vtx_bnd_init(3, vtx_bnd_idx.size()),
            vtx_bnd_next(3, vtx_bnd_idx.size()), vtx_delta;
    vtx_delta = CoordMat3D::Zero(3, vtx_cur.cols());

    auto out_filename = config["out_filename"].get<std::string>();

    int save_cnt = 0;
    auto save = [&]() {
        auto filename = ssprintf("%s-%d.obj", out_filename.c_str(), save_cnt);
        FILE* fout = fopen(filename.c_str(), "w");
        sanm_assert(fout);
        SANM_DEFER(std::bind(::fclose, fout));
        TetrahedralMesh::write_to_file(fout, vtx_cur, mesh->surface_list());
        ++save_cnt;
    };

    json last_stat;
    auto update_to_next = [&](const char* name, bool require_refine) {
        for (size_t i = 0; i < vtx_bnd_idx.size(); ++i) {
            int j = vtx_bnd_idx[i];
            vtx_delta.col(j) = vtx_bnd_next.col(i) - vtx_cur.col(j);
        }
        last_stat = run_with_vtx_delta(name, config, deformable, vtx_delta,
                                       vtx_cur, require_refine);
        save();
    };

    auto copy_to_bnd_init = [&]() {
        for (size_t i = 0; i < vtx_bnd_idx.size(); ++i) {
            int j = vtx_bnd_idx[i];
            vtx_bnd_init.col(i) = vtx_cur.col(j);
        }
    };

    bool save_interm = json_get_bool_opt(config, "save_interm");
    const_cast<json&>(config)["save_interm"] = false;  // disable in rotating
    const double rotate_split =
            json_get_opt<double>(config, "rotate_split", 90);
    fp_t remain_rotate = config["rotate"].get<fp_t>(), finished_rotate = 0;
    save();
    copy_to_bnd_init();
    for (int quadrant_cnt = 0; remain_rotate > 1e-5; ++quadrant_cnt) {
        Eigen::Matrix<fp_t, 3, 3> rmat;
        fp_t rotate = std::min<fp_t>(remain_rotate, rotate_split);
        remain_rotate -= rotate;
        finished_rotate += rotate;
        rotate = finished_rotate * M_PI / 180;
        rmat << 1, 0, 0,                                 //
                0, std::cos(rotate), -std::sin(rotate),  //
                0, std::sin(rotate), std::cos(rotate);   //

        vtx_bnd_next.noalias() = rmat * vtx_bnd_init;
        Eigen::Matrix<fp_t, 3, 1> shift =
                vtx_bnd_init.rowwise().mean() - vtx_bnd_next.rowwise().mean();
        vtx_bnd_next.colwise() += shift;

        update_to_next(ssprintf("rot%d(rem %.1f)", quadrant_cnt, remain_rotate)
                               .c_str(),
                       false);
    }

    copy_to_bnd_init();
    const_cast<json&>(config)["save_interm"] = save_interm;
    for (json bend_desc : config["bend"]) {
        fp_t angle = bend_desc["angle"].get<fp_t>() * M_PI / 180;
        auto shift = json_get_vec3(bend_desc["shift"]);
        Eigen::Matrix<fp_t, 3, 3> rmat;
        rmat << std::cos(angle), -std::sin(angle), 0,  //
                std::sin(angle), std::cos(angle), 0,   //
                0, 0, 1;                               //
        vtx_bnd_next.noalias() = rmat * vtx_bnd_init;
        vtx_bnd_next.colwise() += shift * spacing;
        update_to_next("bend", true);
    }
    last_stat["V"] = mesh->nr_vertices();
    last_stat["F"] = mesh->nr_faces();
    save_json(out_filename + ".json", last_stat);
}

void mesh_twist(const std::filesystem::path& rootpath, const json& config) {
    auto material = make_material_property(config["material"]);
    std::filesystem::path mesh_file_path =
            rootpath / config["mesh"].get<std::string>();
    std::string mesh_file = mesh_file_path;
    auto mesh = TetrahedralMesh::from_tetgen_files(mesh_file);
    if (fp_t s = json_get_opt<double>(config, "scale", 0); s > 0) {
        mesh->resize_inplace(s);
    }
    printf("mesh twist: V=%zu F=%zu\n", mesh->nr_vertices(), mesh->nr_faces());
    TetrahedralDeformableBody deformable{material, mesh};

    Vec3 twist_axis = json_get_vec3(config["axis"]);
    auto out_filename = config["out_filename"].get<std::string>();

    // setup boundary conditions
    std::vector<size_t> vtx_bnd_idx;
    fp_t proj_dist;
    std::unordered_set<int> fixed_vid;
    {
        fp_t proj_min = std::numeric_limits<fp_t>::infinity(),
             proj_max = -proj_min;
        for (size_t i = 0; i < mesh->nr_vertices(); ++i) {
            fp_t p = mesh->vertices().col(i).dot(twist_axis);
            proj_min = std::min(proj_min, p);
            proj_max = std::max(proj_max, p);
        }
        proj_dist = proj_max - proj_min;
        fp_t thresh0 = proj_min +
                       (proj_max - proj_min) * config["ratio_lo"].get<double>(),
             thresh1 =
                     proj_min + (proj_max - proj_min) *
                                        (1 - config["ratio_hi"].get<double>());
        bool include_int_points =
                json_get_bool_opt(config, "include_int_points");
        auto& surface = mesh->surface_vtx();
        sanm_assert(!surface.empty());
        printf("proj range: %g %g thr=%g,%g\n", proj_min, proj_max, thresh0,
               thresh1);
        CoordMask3D& fixed = deformable.coord_fixed_mask();
        for (size_t i = 0; i < mesh->nr_vertices(); ++i) {
            fp_t p = mesh->vertices().col(i).dot(twist_axis);
            if ((p <= thresh0 || p >= thresh1) &&
                (include_int_points || surface.count(i))) {
                for (int j = 0; j < 3; ++j) {
                    fixed(j, i) = true;
                }
                if (p >= thresh1) {
                    vtx_bnd_idx.push_back(i);
                }
            }
        }
        for (size_t i = 0; i < mesh->nr_vertices(); ++i) {
            if (fixed(0, i)) {
                fixed_vid.insert(i);
            }
        }
        save_mesh(out_filename + "-orig.obj", *mesh);
        save_mesh(out_filename + "-boundary.obj", *mesh, &fixed_vid);
    }

    CoordMat3D f_load_full, *f_load_full_ptr = nullptr, vtx_cur;
    if (json_get_bool_opt(config, "add_gravity")) {
        Vec3 g_acc = json_get_vec3(config["g"]);
        fp_t tot_gravity = 0;
        f_load_full = CoordMat3D::Zero(3, mesh->nr_vertices());
        f_load_full_ptr = &f_load_full;
        const auto& volumes = mesh->tet_volumes();
        for (size_t i = 0; i < mesh->nr_tet(); ++i) {
            Vec3 gravity = volumes[i] * material.density() * g_acc;
            tot_gravity += gravity.norm();
            Vec3 node_g = gravity / 4;
            for (int j = 0; j < 4; ++j) {
                f_load_full.col(mesh->tetrahedrons()(j, i)) += node_g;
            }
        }

        printf("add gravity=%.3f\n", tot_gravity);

        auto new_config = config;
        new_config["save_interm"] = false;
        auto mesh_deformed =
                run_and_save("gravity_init", new_config, deformable, false,
                             f_load_full, false);
        save_mesh(out_filename + "-gravity.obj", *mesh_deformed);
        vtx_cur = mesh_deformed->vertices();
    } else {
        vtx_cur = deformable.mesh().vertices();
    }

    CoordMat3D vtx_bnd_next(3, vtx_bnd_idx.size()), vtx_delta;
    for (size_t i = 0; i < vtx_bnd_idx.size(); ++i) {
        int j = vtx_bnd_idx[i];
        vtx_bnd_next.col(i) = vtx_cur.col(j);
    }

    auto apply_trans = [&](const json& config) {
        fp_t angle = config["angle"].get<fp_t>() * M_PI / 180;
        auto shift = json_get_vec3(config["shift"]);
        Eigen::Matrix<fp_t, 3, 3> rmat;
        auto rot_axis = json_get_opt<int>(config, "rot_axis", 2);
        rmat.setIdentity();
        {
            Eigen::Matrix<fp_t, 2, 2> rmat_small;
            rmat_small << std::cos(angle), -std::sin(angle),  //
                    std::sin(angle), std::cos(angle);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (i != rot_axis && j != rot_axis) {
                        rmat(i, j) = rmat_small(i - (i > rot_axis),
                                                j - (j > rot_axis));
                    }
                }
            }
        }
        vtx_bnd_next = (rmat * vtx_bnd_next).eval();
        vtx_bnd_next.colwise() += shift * proj_dist;
    };

    if (config.contains("transforms")) {
        for (auto&& i : config["transforms"]) {
            apply_trans(i);
        }
    } else {
        apply_trans(config);
    }

    vtx_delta = CoordMat3D::Zero(3, vtx_cur.cols());
    for (size_t i = 0; i < vtx_bnd_idx.size(); ++i) {
        int j = vtx_bnd_idx[i];
        vtx_delta.col(j) = vtx_bnd_next.col(i) - vtx_cur.col(j);
    }

    {
        TetrahedralMesh mesh_copy{*mesh};
        mesh_copy.replace_vtx(vtx_cur + vtx_delta);
        save_mesh(out_filename + "-boundary-dst.obj", mesh_copy, &fixed_vid);
    }

    auto stat = run_with_vtx_delta("mesh_twist", config, deformable, vtx_delta,
                                   vtx_cur, false, f_load_full_ptr);
    mesh->replace_vtx(vtx_cur);  // note that mesh is modified inplace here
    save_mesh(out_filename + ".obj", *mesh);
    save_json(out_filename + ".json", stat);
    save_out_surface_vtx(config, *mesh);
}

void setup_boundary_by_config(TetrahedralDeformableBody& deformable,
                              const Vec3& default_proj_dir,
                              const json& config) {
    Vec3 proj_dir;
    fp_t thresh;
    const auto& mesh = deformable.mesh();
    auto& vtx = mesh.vertices();
    {
        fp_t proj_min = std::numeric_limits<fp_t>::infinity(),
             proj_max = -proj_min;
        if (config.contains("boundary_proj_dir")) {
            proj_dir = json_get_vec3(config["boundary_proj_dir"]);
        } else {
            proj_dir = default_proj_dir;
        }
        proj_dir.normalize();
        for (size_t i = 0; i < mesh.nr_vertices(); ++i) {
            fp_t p = vtx.col(i).dot(proj_dir);
            proj_min = std::min(proj_min, p);
            proj_max = std::max(proj_max, p);
        }
        thresh = proj_min +
                 (proj_max - proj_min) * config["boundary_thresh"].get<fp_t>();
        printf("proj range: %g %g thr=%g\n", proj_min, proj_max, thresh);
    }

    std::function<bool(const Vec3&)> filter;
    if (config.contains("boundary_filter")) {
        auto fcfg = config["boundary_filter"];
        auto dir = json_get_vec3(fcfg["dir"]);
        fp_t proj_min = std::numeric_limits<fp_t>::infinity(),
             proj_max = -proj_min;
        for (size_t i = 0; i < mesh.nr_vertices(); ++i) {
            fp_t p = vtx.col(i).dot(dir);
            proj_min = std::min(proj_min, p);
            proj_max = std::max(proj_max, p);
        }
        fp_t d = proj_max - proj_min,
             th0 = proj_min + d * fcfg["min"].get<fp_t>(),
             th1 = proj_min + d * fcfg["max"].get<fp_t>();
        printf("filter range: [%g, %g]\n", th0, th1);
        filter = [th0, th1, dir](const Vec3& v) {
            fp_t p = v.dot(dir);
            return p >= th0 && p <= th1;
        };
    }

    auto& surface = mesh.surface_vtx();
    sanm_assert(!surface.empty());
    CoordMask3D& fixed = deformable.coord_fixed_mask();
    for (size_t i = 0; i < mesh.nr_vertices(); ++i) {
        fp_t p = vtx.col(i).dot(proj_dir);
        if (p <= thresh && surface.count(i)) {
            if (filter && !filter(vtx.col(i))) {
                continue;
            }
            for (int j = 0; j < 3; ++j) {
                fixed(j, i) = true;
            }
        }
    }
}

void gravity(const std::filesystem::path& rootpath, const json& config) {
    auto material = make_material_property(config["material"], true);
    std::filesystem::path mesh_file_path =
            rootpath / config["mesh"].get<std::string>();
    std::string mesh_file = mesh_file_path;
    auto mesh = TetrahedralMesh::from_tetgen_files(mesh_file);
    TetrahedralDeformableBody deformable{material, mesh};
    size_t nr_fixed = 0;
    Vec3 g_acc = json_get_vec3(config["g"]);

    if (config.contains("scale")) {
        mesh->resize_inplace(config["scale"].get<fp_t>());
    }

    // setup boundary conditions
    {
        std::ifstream fin{mesh_file + ".bou"};
        CoordMask3D& fixed = deformable.coord_fixed_mask();
        if (fin.good()) {
            for (int idx; fin >> idx;) {
                sanm_assert(idx > 0);
                --idx;
                for (int i = 0; i < 3; ++i) {
                    fixed(i, idx) = true;
                }
            }
        } else {
            printf("bou file does not exist; fix lowest points ...\n");
            setup_boundary_by_config(deformable, -g_acc, config);
        }
        std::unordered_set<int> fixed_vid;
        for (size_t i = 0; i < mesh->nr_vertices(); ++i) {
            if (fixed(0, i)) {
                ++nr_fixed;
                fixed_vid.insert(i);
            }
        }
        save_mesh(config["out_filename"].get<std::string>() + "-boundary.obj",
                  *mesh, &fixed_vid);
    }

    // setup gravity
    fp_t tot_gravity = 0;
    CoordMat3D f_load_full = CoordMat3D::Zero(3, mesh->nr_vertices());
    const auto& volumes = mesh->tet_volumes();
    for (size_t i = 0; i < mesh->nr_tet(); ++i) {
        Vec3 gravity = volumes[i] * material.density() * g_acc;
        tot_gravity += gravity.norm();
        Vec3 node_g = gravity / 4;
        for (int j = 0; j < 4; ++j) {
            f_load_full.col(mesh->tetrahedrons()(j, i)) += node_g;
        }
    }

    printf("mesh loading finished %s:\n"
           " nr_vtx=%zu nr_tet=%zu boundary_vtx=%zu gravity=%.3f\n",
           mesh_file.c_str(), mesh->nr_vertices(), mesh->nr_tet(), nr_fixed,
           tot_gravity);

    run_and_save(ssprintf("mesh %s", mesh_file_path.filename().c_str()).c_str(),
                 config, deformable, json_get_bool_opt(config, "inverse"),
                 f_load_full);
}

json read_json(const char* fpath) {
    std::ifstream fin{fpath};
    sanm_assert(fin.good(), "failed to open %s", fpath);
    std::string data{std::istreambuf_iterator<char>{fin}, {}};
    return json::parse(data);
}

void init_global_config(const char* fpath) {
    auto config = read_json(fpath);
    sanm::SparseSolver::set_verbosity(config["verbosity"].get<int>());
    sanm::set_num_threads(config["threads"].get<int>());
    if (config.contains("sparse_solver_threads")) {
        int n = config["sparse_solver_threads"].get<int>();
        SparseSolver::set_num_threads(n);
    }
}
}  // anonymous namespace

int do_main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <system config file> <task config file> "
                "[<task override json files ...>]\n",
                argv[0]);
        return -1;
    }
    init_global_config(argv[1]);
    json config = read_json(argv[2]);
    for (int i = 3; i < argc; ++i) {
        json extra = read_json(argv[i]);
        config.update(extra);
    }
    auto func = config["func"].get<std::string>();
    if (func == "test_single_tet_inverse") {
        test_single_tet_inverse(config);
        return 0;
    }
    if (func == "test_cuboid") {
        test_cuboid(config);
        return 0;
    }
    if (func == "test_cuboid_twist") {
        test_cuboid_twist(config);
        return 0;
    }
    if (func == "gravity") {
        gravity(std::filesystem::path{argv[2]}.parent_path(), config);
        return 0;
    }
    if (func == "mesh_twist") {
        mesh_twist(std::filesystem::path{argv[2]}.parent_path(), config);
        return 0;
    }
    throw std::runtime_error{ssprintf("unknown func: %s", func.c_str())};
}

int main(int argc, char** argv) {
    int retval = 0;
    try {
        retval = do_main(argc, argv);
    } catch (std::exception& e) {
        fprintf(stderr, "caught exception: %s\n", e.what());
        retval = 2;
    }

    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        printf("memory: %.3fGiB\n", usage.ru_maxrss / (1024.0 * 1024));
    }

    return retval;
}
