/**
 * \file tests/symbolic.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/analytic_unary.h"
#include "libsanm/anm.h"
#include "libsanm/oprs.h"
#include "tests/helper.h"

#include <array>
#include <cstring>
#include <functional>

using namespace sanm;
using namespace test;
using namespace symbolic;

namespace {

TensorND batched_mul_eye(const TensorND& x, size_t dim) {
    return TensorND{}.as_batched_diag_scalar(x, dim);
}
SymbolVar batched_mul_eye(SymbolVar x, size_t dim) {
    return x.batched_mul_eye(dim);
}

//! set \p maxiter as a negative number to break rather than fail
TensorND run_anm(ANMSolverVecScale& solver, fp_t t_dst, int maxiter = 20) {
    fp_t t_upper;
    int iter = 0;
    for (;;) {
        t_upper = solver.get_t_upper();
        printf(" %.3f", t_upper);
        fflush(stdout);
        ++iter;
        if (maxiter < 0) {
            if (iter >= -maxiter) {
                t_dst = t_upper;
                break;
            }
        } else {
            REQUIRE(iter <= maxiter);
        }
        if (t_upper > t_dst) {
            break;
        }
        solver.update_approx();
    }
    printf(" (%d)\n", iter);
    auto solved = solver.eval(solver.solve_a(t_dst));
    REQUIRE(solved.second == Approx(t_dst));
    return solved.first;
}

//! solve f(x)=y, and return the solution
TensorND anm_general_solve(const char* name, SymbolVar f, const TensorND& x0,
                           const TensorND& y, int maxiter = 20,
                           const ANMEqnSolver::HyperParam& hyper_param = {}) {
    auto id_remap = SparseLinearDesc::make_identity(x0.shape());
    ANMEqnSolver anm_solver{f.node(), id_remap, id_remap, x0, -y, hyper_param};
    printf("%s:", name);
    int iter = 0;
    while (!anm_solver.converged()) {
        printf(" %.2g", anm_solver.residual_rms());
        fflush(stdout);
        ++iter;
        REQUIRE(iter <= maxiter);
        anm_solver.next_iter();
    }
    printf(" (%d):%g\n", iter, anm_solver.residual_rms());
    return anm_solver.get_x();
}

//! solve f(x)=y, and return the solution; do not incorporate error correction
TensorND anm_general_solve_strict(const char* name, SymbolVar f,
                                  const TensorND& x0, const TensorND& y,
                                  int maxiter = 20) {
    TensorND f0 = eval_unary_func(f.node(), x0);
    SymbolVar y_off = linear_combine(
            {{1., f}, {-1., constant(*f.node()->owner_graph(), f0)}});
    auto id_remap = SparseLinearDesc::make_identity(x0.shape());
    ANMSolverVecScale anm_solver{y_off.node(), id_remap, id_remap,
                                 x0,           0,        f0 - y};
    printf("%s:", name);
    return run_anm(anm_solver, 1, maxiter);
}

void check_taylor_prop(SymbolVar y, const TensorArray& xarr,
                       std::span<const fp_t> avals, fp_t eps_coeff,
                       fp_t eps_eval, fp_t max_diff_eval,
                       const std::string& plot_filename = {}) {
    TaylorCoeffProp tprop{y.node(), true};
    TensorArray yarr;
    yarr.emplace_back(tprop.push_xi(xarr[0]));
    auto& jacobian = tprop.get_jacobian(tprop.inputs()[0]);
    for (size_t i = 1; i < xarr.size(); ++i) {
        auto& xi = xarr[i];
        auto& bi = tprop.compute_next_order_bias();
        auto yi_linear = jacobian.apply(xi).reshape(yarr[0].shape()) + bi;
        auto& yi_comp = tprop.push_xi(xi);
        require_tensor_eq(yi_linear, yi_comp, {eps_coeff, eps_coeff});
        yarr.emplace_back(yi_comp);
    }

    if (!plot_filename.empty()) {
        FILE* fout = fopen((plot_filename + "_term_lognorm.txt").c_str(), "w");
        REQUIRE(fout);
        for (size_t i = 0; i < yarr.size(); ++i) {
            fprintf(fout, "%zu %g\n", i, std::log(yarr[i].norm_l2()));
        }
        fclose(fout);

        fout = fopen((plot_filename + "_approx_logerr.txt").c_str(), "w");
        REQUIRE(fout);
        fp_t amax = avals.back() * 3, amin = -amax;
        const int N = 100;
        for (int i = 0; i < N; ++i) {
            fp_t a = (amax - amin) / N * i + amin;
            auto xt = unary_polynomial::eval_tensor(xarr, a),
                 yt0 = unary_polynomial::eval_tensor(
                         {yarr.begin(), yarr.begin() + yarr.size() / 2}, a),
                 yt1 = unary_polynomial::eval_tensor(yarr, a),
                 yget = eval_unary_func(y.node(), xt);
            fp_t err0 = (yt0 - yget).norm_l2(), err1 = (yt1 - yget).norm_l2();
            fprintf(fout, "%g %g %g\n", a, std::log(err0), std::log(err1));
        }
        fclose(fout);
    }

    for (fp_t a : avals) {
        auto xt = unary_polynomial::eval_tensor(xarr, a),
             yt = unary_polynomial::eval_tensor(yarr, a),
             yget = eval_unary_func(y.node(), xt);
        require_tensor_eq(yt, yget, {eps_eval, eps_eval}, max_diff_eval);
    }
}
}  // anonymous namespace

TEST_CASE("Symbolic.Pow") {
    /*
     * let e = exp_n / exp_d
     * let v = pow(x0, e)
     * known: pow(x0, e) + 1 * v = 0
     * solve x where pow(x, e) + 2 * v = 0
     */
    auto run = [&](fp_t exp, fp_t xlow = 1., fp_t xhigh = 2.) {
        ComputingGraph cg;
        SymbolVar x = placeholder(cg);
        TensorRNG rng{xlow, xhigh};

        TensorND x0 = rng({3, 2});

        SymbolVar xpow = x.pow(exp);
        auto id_remap = SparseLinearDesc::make_identity(x0.shape());
        ANMSolverVecScale anm_solver{xpow.node(), id_remap, id_remap,
                                     x0,          1,        -x0.pow(exp)};

        printf("ANM pow(%g):", exp);
        auto sol = run_anm(anm_solver, 2);
        require_tensor_eq(sol, x0 * std::pow(fp_t(2), fp_t(1) / exp));
    };

    SECTION("square") { run(2); }
    SECTION("fractional") { run(5. / 3.); }
    SECTION("inv") { run(-1); }
    SECTION("inv-fractional") { run(-2.4); }
    SECTION("rsqrt") { run(-.5); }
    SECTION("sqrt") { run(.5); }
    SECTION("smallint") {
        for (int i = -5; i <= 5; ++i) {
            if (i && i != 1) {
                run(i, -5, 5);
            }
        }
    }
}

TEST_CASE("Symbolic.MatInvMul") {
    ComputingGraph cg;
    SymbolVar x = placeholder(cg);
    TensorRNG rng{1.f, 4.f};

    TensorND x0 = rng({9, 4, 4});
    {
        auto xptr = x0.rwptr();
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 4; ++j) {
                xptr[(i * 4 + j) * 4 + j] += 4;
            }
        }
    }
    auto id_remap = SparseLinearDesc::make_identity(x0.shape());

    TensorND x0inv = -x0.batched_matinv();

    SECTION("simple-inv") {
        for (bool is_left : {false, true}) {
            SymbolVar xinv = batched_mat_inv_mul(x, {}, is_left);
            ANMSolverVecScale anm_solver{xinv.node(), id_remap, id_remap,
                                         x0,          1,        x0inv};

            printf(is_left ? "simple-inv-l:" : "simple-inv-r:");
            auto sol = run_anm(anm_solver, 2);
            require_tensor_eq(sol, x0 * 0.5);
        }
    }

    SymbolVar a = x.pow(1.5);
    SECTION("left") {
        SymbolVar xinv = batched_mat_inv_mul(x, a, true);
        ANMSolverVecScale anm_solver{
                xinv.node(), id_remap,
                id_remap,    x0,
                1,           TensorND{}.as_batched_mm(x0.pow(1.5), x0inv)};

        printf("left-inv:");
        auto sol = run_anm(anm_solver, 1.3);
        require_tensor_eq(sol, x0 * (1.3 * 1.3));
    }

    SECTION("right") {
        SymbolVar xinv = batched_mat_inv_mul(x, a, false);
        ANMSolverVecScale anm_solver{
                xinv.node(), id_remap,
                id_remap,    x0,
                1,           TensorND{}.as_batched_mm(x0inv, x0.pow(1.5))};

        printf("right-inv:");
        auto sol = run_anm(anm_solver, 1.2);
        require_tensor_eq(sol, x0 * (1.2 * 1.2));
    }
}

TEST_CASE("Symbolic.ElemArith") {
    ComputingGraph cg;
    SymbolVar x = placeholder(cg);
    TensorRNG rng{2., 5.};

    TensorND x0 = rng({9, 4, 4});
    auto id_remap = SparseLinearDesc::make_identity(x0.shape());
    SECTION("add") {
        TensorND x0inv = x0.batched_matinv();
        SymbolVar y = x + x.pow(1.5);
        TensorND y0 = -(x0 + x0.pow(1.5));
        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, y0};

        printf("add:");
        auto sol = run_anm(anm_solver, 2);
        require_tensor_eq(sol + sol.pow(1.5), y0 * (-2));
    }

    SECTION("sub") {
        TensorND x0inv = x0.batched_matinv();
        SymbolVar y = x.pow(5. / 3.) - x.pow(1.5);
        TensorND y0 = -(x0.pow(5. / 3.) - x0.pow(1.5));
        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, y0};

        printf("sub:");
        auto sol = run_anm(anm_solver, 2);
        require_tensor_eq(sol.pow(5. / 3.) - sol.pow(1.5), y0 * (-2));
    }

    SECTION("mul") {
        TensorND x0inv = x0.batched_matinv();
        SymbolVar y = x.pow(-.75) * x.pow(1.5);
        TensorND y0 = -(x0.pow(-0.75) * x0.pow(1.5));
        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, y0};

        printf("mul:");
        auto sol = run_anm(anm_solver, 1.42);
        require_tensor_eq(sol.pow(-0.75) * sol.pow(1.5), y0 * (-1.42));
    }

    SECTION("mul-bcast") {
        auto f = [](auto x) { return x.pow(2.3) * x.reduce_sum(-1); };
        SymbolVar y = f(x);
        TensorND y0 = -f(x0);
        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, y0};

        printf("mul-bcast:");
        auto sol = run_anm(anm_solver, 2);
        require_tensor_eq(f(sol), y0 * (-2));
    }

    SECTION("mul-bcast-fullgy") {
        auto f = [](auto x) {
            auto xs = x.reduce_sum(-1);
            return (x.pow(1.2) * xs + batched_mul_eye(xs, 4)).batched_matinv();
        };
        SymbolVar y = f(x);
        TensorND y0 = -f(x0);
        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, y0};

        printf("mul-bcast-fullgy:");
        auto sol = run_anm(anm_solver, 2);
        require_tensor_eq(f(sol), y0 * (-2));
    }
}

TEST_CASE("Symbolic.LinearCombination") {
    ComputingGraph cg;
    SymbolVar x = placeholder(cg),
              y = linear_combine({{1.2_fp, x.reduce_sum(-1)},
                                  {2.3_fp, x.pow(2. / 3.)},
                                  {1.4_fp, x.pow(1.5)}},
                                 2.5);
    auto f = [](const TensorND& x) {
        return x.pow(1.5) * 1.4 + x.pow(2.0 / 3) * 2.3 +
               x.reduce_sum(-1, true) * 1.2 +
               x.make_same_shape().fill_with_inplace(2.5);
    };
    TensorRNG rng{2., 5.};

    TensorND x0 = rng({9, 4, 4});
    auto id_remap = SparseLinearDesc::make_identity(x0.shape());
    auto y0 = f(x0);
    ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, -y0};
    printf("linear-combine:");
    auto sol = run_anm(anm_solver, 2);
    require_tensor_eq(f(sol), y0 * 2);
}

TEST_CASE("Symbolic.Determinant") {
    ComputingGraph cg;
    SymbolVar x = placeholder(cg);
    TensorRNG rng;

    fp_t solve_dst = 2;

    auto run = [&](const char* name, size_t batch, size_t mdim) {
        TensorND x0 = rng({batch, mdim, mdim}),
                 y0 = x0.batched_determinant() * x0;
        auto id_remap = SparseLinearDesc::make_identity(x0.shape());
        SymbolVar y = x.batched_det() * x;

        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, -y0};
        printf("%s:", name);
        Timer timer;
        timer.start();
        auto sol = run_anm(anm_solver, solve_dst);
        timer.stop();
        require_tensor_eq(sol.batched_determinant() * sol, y0 * solve_dst);
        return timer.time();
    };

    SECTION("small") { run("batched_det-small", 1, 3); }
    SECTION("mid") { run("batched_det-mid", 10, 4); }
    SECTION("large0") { run("batched_det-large0", 10, 5); }
    SECTION("large1") { run("batched_det-large1", 10, 7); }
    SECTION("bench") {
#if __OPTIMIZE__
        solve_dst = 1.4;
        auto time = run("batched_det-xlarge", 10000, 3);
        printf("batched_det-xlarge: solve time=%.3fs\n", time);
#else
        printf("batched_det-xlarge: disabled due to no optmization\n");
#endif
    }
}

TEST_CASE("Symbolic.Reduce") {
    ComputingGraph cg;
    SymbolVar x = placeholder(cg);
    TensorRNG rng;

    auto run = [&](const char* name, size_t batch, size_t dim0, size_t dim1,
                   int axis) {
        TensorShape x0shp;
        if (dim1) {
            x0shp = {batch, dim0, dim1};
        } else {
            x0shp = {batch, dim0};
        }
        TensorND x0 = rng(x0shp), y0 = x0.reduce_sum(axis, true) * x0.pow(-2);
        auto id_remap = SparseLinearDesc::make_identity(x0.shape());
        SymbolVar y = x.reduce_sum(axis, true) * x.pow(-2);

        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, -y0};
        printf("%s:", name);
        auto sol = run_anm(anm_solver, 2);
        require_tensor_eq(sol.reduce_sum(axis, true) * sol.pow(-2), y0 * 2);
    };

    SECTION("axis") { run("reduce-axis", 10, 5, 0, 1); }
    SECTION("flatten") { run("reduce-flatten", 8, 9, 7, -1); }
}

TEST_CASE("Symbolic.Transpose") {
    ComputingGraph cg;
    SymbolVar x = placeholder(cg), y = x.pow(1.5).batched_transpose();
    TensorRNG rng{1., 2.};

    auto f = [](const TensorND& x) { return x.pow(1.5).batched_transpose(); };

    TensorND x0 = rng({5, 4, 6}), y0 = f(x0);
    ANMSolverVecScale anm_solver{y.node(),
                                 SparseLinearDesc::make_identity(x0.shape()),
                                 SparseLinearDesc::make_identity(y0.shape()),
                                 x0,
                                 1,
                                 -y0};
    printf("trans:");
    auto sol = run_anm(anm_solver, 2);
    require_tensor_eq(f(sol), y0 * 2);
}

TEST_CASE("Symbolic.TransMul") {
    ComputingGraph cg;
    SymbolVar x = placeholder(cg),
              y = x.batched_matmul(x.batched_transpose()).reduce_sum(-1) * x;
    TensorRNG rng;

    auto f = [](const TensorND& x) {
        TensorND t;
        t = t.as_batched_mm(x, x.batched_transpose()).reduce_sum(-1, true);
        return x * t;
    };

    TensorND x0 = rng({5, 4, 6}), y0 = f(x0);
    auto id_remap = SparseLinearDesc::make_identity(x0.shape());
    ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, -y0};
    printf("trans-mul:");
    auto sol = run_anm(anm_solver, 2);
    require_tensor_eq(f(sol), y0 * 2);
}

namespace {
class RandSparseLinearDesc final : public SparseLinearDesc {
    std::vector<std::vector<InputElem>> m_entries;
    TensorShape m_shp_in, m_shp_out;

public:
    RandSparseLinearDesc(const TensorShape& shp_in, const TensorShape& shp_out,
                         Xorshift128pRng& rng)
            : m_shp_in{shp_in}, m_shp_out{shp_out} {
        size_t nr_in = shp_in.total_nr_elems(),
               nr_out = shp_out.total_nr_elems(),
               entry_size = nr_out > nr_in ? 2 : nr_in / nr_out + 1;
        m_entries.resize(nr_out);
        std::uniform_real_distribution<fp_t> udist{-1., 1.};
        std::vector<size_t> inp_loc_perm(nr_in);
        for (size_t i = 0; i < nr_in; ++i) {
            inp_loc_perm[i] = i;
        }
        size_t inp_loc_perm_read = nr_in;
        auto next_inp_loc = [&]() {
            if (inp_loc_perm_read == nr_in) {
                std::shuffle(inp_loc_perm.begin(), inp_loc_perm.end(), rng);
                inp_loc_perm_read = 0;
            }
            return inp_loc_perm[inp_loc_perm_read++];
        };
        for (auto& i : m_entries) {
            if (nr_out > nr_in && !empty_used &&
                (rng() % 9 == 0 || (&i == &m_entries.back()))) {
                // test empty
                empty_used = true;
                continue;
            }

            if (rng() % 5 == 0) {
                // test identical input
                i.resize(2);
                i[0].idx = i[1].idx = next_inp_loc();
                identical_used = true;
            } else {
                i.resize(entry_size);
                for (auto& j : i) {
                    j.idx = next_inp_loc();
                }
            }
            for (auto& j : i) {
                j.coeff = udist(rng);
            }
        }
    }

    TensorShape out_shape() const override { return m_shp_out; }

    TensorShape inp_shape() const override { return m_shp_in; }

    Linear1d get(size_t dst_index, size_t) const override {
        return m_entries.at(dst_index);
    }

    bool empty_used = false;
    bool identical_used = false;
};
}  // anonymous namespace

TEST_CASE("Symbolic.IORemap") {
    ComputingGraph cg;
    SymbolVar xcg = placeholder(cg), ycg = xcg.pow(2);
    TensorRNG rng;

    auto run = [&](const char* name, const TensorShape& xshp,
                   const TensorShape& midshp) {
        auto remap_in = std::make_shared<RandSparseLinearDesc>(xshp, midshp,
                                                               rng.raw_rng()),
             remap_out = std::make_shared<RandSparseLinearDesc>(midshp, xshp,
                                                                rng.raw_rng());
        auto f = [&](const TensorND& x) {
            auto xcg = remap_in->apply(x), ycg = xcg.pow(2);
            return remap_out->apply(ycg);
        };
        auto x0 = rng(xshp), y0 = f(x0);

        ANMSolverVecScale anm_solver{ycg.node(), remap_in, remap_out,
                                     x0,         1,        -y0};
        printf("ioremap-%s:", name);
        auto sol = run_anm(anm_solver, 2);
        require_tensor_eq(f(sol), y0 * 2);
        REQUIRE(remap_in->empty_used);
        REQUIRE(remap_in->identical_used);
        REQUIRE_FALSE(remap_out->empty_used);
        REQUIRE(remap_out->identical_used);
    };

    SECTION("small") { run("small", {2, 2}, {4, 4}); }

    SECTION("large") { run("large", {5, 11}, {10, 4, 6}); }
}

TEST_CASE("Symbolic.Constant") {
    TensorRNG rng;
    const size_t batch = 5, dim = 4;
    TensorND x0 = rng({batch, dim, dim}), cval = rng({batch, 1}),
             eye{{batch, dim, dim}};
    {
        auto eptr = eye.woptr();
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    eptr[(i * dim + j) * dim + k] = (j == k);
                }
            }
        }
    }

    ComputingGraph cg;
    SymbolVar x = placeholder(cg),
              y = (x.reduce_sum(-1).batched_mul_eye(4) + x) *
                  symbolic::constant(cg, cval);

    auto f = [&eye, &cval](const TensorND& x) {
        return (x.reduce_sum(-1, true) * eye + x) * cval;
    };

    auto id_remap = SparseLinearDesc::make_identity(x0.shape());
    auto y0 = f(x0);
    ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, -y0};
    printf("constant:");
    auto sol = run_anm(anm_solver, 2);
    require_tensor_eq(f(sol), y0 * 2);
}

TEST_CASE("Symbolic.Analytic") {
    auto run = [&](const char* name, UnaryAnalyticTraitPtr trait, fp_t xlow,
                   fp_t xhigh, std::function<SymbolVar(SymbolVar)> impl_sym,
                   fp_t t_dst = 2) {
        auto f = [&trait](const TensorND& x) { return trait->eval(x); };

        TensorRNG rng{xlow, xhigh};
        TensorND x0 = rng({10, 20});

        ComputingGraph cg;
        SymbolVar x = placeholder(cg), y = impl_sym(x);
        auto id_remap = SparseLinearDesc::make_identity(x0.shape());
        auto y0 = f(x0);
        ANMSolverVecScale anm_solver{y.node(), id_remap, id_remap, x0, 1, -y0};
        printf("%s:", name);
        auto sol = run_anm(anm_solver, t_dst);
        require_tensor_eq(f(sol), y0 * t_dst);
    };

    SECTION("ln") {
        run("ln", UnaryAnalyticTrait::make_log(), 0.1, 2.5,
            [](SymbolVar x) { return x.log(); });
    }
}

TEST_CASE("Symbolic.GeneralSolve") {
    SECTION("sqr") {
        auto f = [](auto x) { return x * x; };

        TensorRNG rng{0.2, 1.5};
        TensorND xsol = rng({10, 20}), ysol = f(xsol),
                 xinit = xsol * rng(xsol.shape(), 0.6, 1.5);

        ComputingGraph cg;
        auto xt = anm_general_solve("sqr", f(placeholder(cg)), xinit, ysol);
        require_tensor_eq(f(xt), ysol);
    }

    SECTION("pow-log-pow") {
        auto f = [](auto x) { return x.pow(2.3).log().pow(1.5); };

        TensorRNG rng{1.5, 4.3};
        TensorND xsol = rng({10, 20}), ysol = f(xsol),
                 xinit = xsol * rng(xsol.shape(), 0.6, 1.5);

        ComputingGraph cg;
        auto xt = anm_general_solve_strict("pow-log-pow", f(placeholder(cg)),
                                           xinit, ysol);
        require_tensor_eq(f(xt), ysol);
    }

    // pow with zero grad
    auto run_pow_zg = [](int exp) {
        auto f = [exp](auto x) { return x.pow(1.7) + x.log().pow(exp); };

        TensorRNG rng{0.8, 1.5};
        TensorND xsol = rng({10, 8, 3}), ysol = f(xsol),
                 xinit = xsol * rng(xsol.shape(), 0.8, 1.2);
        xsol.rwptr()[2] = 1.3;
        xinit.rwptr()[2] = 1;  // test zero grad of pow

        ComputingGraph cg;
        auto xt = anm_general_solve(ssprintf("pow-zg(%d)", exp).c_str(),
                                    f(placeholder(cg)), xinit, ysol);
        require_tensor_eq(f(xt), ysol);
    };

    SECTION("pow-zg") {
        for (int i : {2, 5, 6, 8, 15}) {
            run_pow_zg(i);
        }
    }

    set_num_threads(2);
    SECTION("pow-zg-mt") {
        for (int i : {2, 5, 6, 8, 15}) {
            run_pow_zg(i);
        }
    }
    set_num_threads(1);
}

TEST_CASE("Symbolic.LogDetTaylorProp") {
    TensorRNG rng;
    TensorArray xarr;

    ComputingGraph cg;
    SymbolVar x = placeholder(cg),
              y = x.batched_transpose().batched_matmul(x).batched_det().log();

    for (int i = 0; i < 5; ++i) {
        xarr.emplace_back(rng({10, 4, 3}));
    }
    for (int i = 0; i < 8; ++i) {
        xarr.emplace_back(xarr[0].fill_with(0));
    }
    check_taylor_prop(y, xarr, std::vector<fp_t>{0.01, -0.01, 0.1}, 1e-4, 1e-3,
                      1e-5, "log_det");
}

TEST_CASE("Symbolic.PolarDecompTaylorProp") {
    TensorRNG rng;
    constexpr size_t batch = 7, n = 4;
    TensorArray xarr;

    ComputingGraph cg;
    SymbolVar x = placeholder(cg), y = x - x.batched_svd_w(true)[2];

    for (int i = 0; i < 5; ++i) {
        xarr.emplace_back(rng({batch, n, n}));
    }
    xarr[0] = unary_polynomial::eval_tensor(xarr, 0.03);
    for (int i = 0; i < 16; ++i) {
        xarr.emplace_back(xarr[0].fill_with(0));
    }
    check_taylor_prop(y, xarr, std::vector<fp_t>{0.01, -0.01, 0.02}, 1e-4, 1e-3,
                      1e-5, "polar_decomp");
}

TEST_CASE("Symbolic.PolarDecompSolve") {
    TensorRNG rng;
    constexpr size_t batch = 7, n = 4;
    auto x0 = rng({batch, n, n}), dx = rng(x0.shape(), -0.05, 0.05);

    auto f = [](auto x, bool require_rotation) {
        return x - x.batched_svd_w(require_rotation)[2];
    };

    auto run = [&](const char* name, bool require_rotation) {
        auto xsol = x0 + dx, ysol = f(xsol, require_rotation);
        ComputingGraph cg;
        auto xt = anm_general_solve(name, f(placeholder(cg), require_rotation),
                                    x0, ysol, 40);
        require_tensor_eq(xsol, xt);
    };

    auto make_eq_singular = [](const TensorND& x, bool require_rotation) {
        auto svd = x.batched_svd_w(require_rotation);
        auto ps = svd[1].rwptr();
        for (size_t i = 0; i < n; ++i) {
            ps[i * n + 1] = ps[i * n];
        }
        return svd[0]
                .batched_mm(TensorND{}.as_batched_diag(svd[1]))
                .batched_mm(svd[0].batched_transpose())
                .batched_mm(svd[2]);
    };

    SECTION("simple") { run("simple", false); }

    SECTION("simple-rot") { run("simple-rot", true); }

    SECTION("eq-singular-x0") {
        auto x0orig = x0;
        x0 = make_eq_singular(x0orig, false);
        run("eqs-x0", false);
        x0 = make_eq_singular(x0orig, true);
        run("eqs-x0-rot", true);
    }

    // XXX: very slow convergence
#if 0
    SECTION("eq-singular-sol") {
        auto xt = x0 + dx;
        x0 = make_eq_singular(xt, false) - dx;
        run("eqs-xt", false);
        x0 = make_eq_singular(xt, true) - dx;
        run("eqs-xt-rot", true);
    }
#endif
}

TEST_CASE("Symbolic.Rosenbrock") {
    auto rosen_der = [](SymbolVar x) {
        SymbolVar xm = x.slice(1, 1, -1), xm_m1 = x.slice(1, None, -2),
                  xm_p1 = x.slice(1, 2, None), x0 = x.slice(1, 0, 1),
                  x1 = x.slice(1, 1, 2), xp1 = x.slice(1, -1, None),
                  xp2 = x.slice(1, -2, -1),
                  der0 = linear_combine(
                          {{-400._fp, x0 * (x1 - x0.pow(2))}, {2._fp, x0}}, -2),
                  der1 = linear_combine({{200._fp, xm},
                                         {-200._fp, xm_m1.pow(2)},
                                         {-400._fp, (xm_p1 - xm.pow(2)) * xm},
                                         {2._fp, xm}},
                                        -2),
                  der2 = linear_combine(
                          {{200._fp, xp1}, {-200._fp, xp2.pow(2)}});
        return concat(std::vector<SymbolVar>{der0, der1, der2}, 1);
    };

    auto vec2tensor = [](const std::vector<fp_t>& x) {
        TensorND y{{1, x.size()}};
        memcpy(y.woptr(), x.data(), x.size() * sizeof(fp_t));
        return y;
    };

    // compare with scipy example
    // https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize
    auto x0 = vec2tensor({1.3, 0.7, 0.8, 1.9, 1.2});

    SECTION("der_correctness") {
        auto der_correct = vec2tensor({515.4, -285.4, -341.6, 2085.4, -482.});
        ComputingGraph cg;
        auto der_get = eval_unary_func(rosen_der(placeholder(cg)).node(), x0);
        require_tensor_eq(der_correct, der_get);
    }

    // it does not work for Rosenbrock, probably because f'(x)= k f'(x0) does
    // not always have a solution for 0 <= k <= 1
    SECTION("minimize") {
        ComputingGraph cg;
        anm_general_solve_strict("rosenbrock",
                                 rosen_der(placeholder(cg)).node(), x0,
                                 x0.fill_with(0), -10);
    }
}

namespace {
class ANMImplicitSolverInputTrans final : public SparseLinearDesc {
    TensorND m_dx;
    mutable InputElem m_ret[2];

public:
    explicit ANMImplicitSolverInputTrans(TensorND dx) : m_dx{std::move(dx)} {
        REQUIRE(m_dx.rank() == 1);
        m_ret[0].coeff = 1;
        m_ret[1].idx = m_dx.shape(0);
    }

    TensorShape out_shape() const override { return m_dx.shape(); }

    TensorShape inp_shape() const override { return {m_dx.shape(0) + 1}; }

    void init_multi_thread(size_t nr_thread) override {
        REQUIRE(nr_thread == 1);
    }

    Linear1d get(size_t dst_index, size_t thread_id) const override {
        m_ret[0].idx = dst_index;
        m_ret[1].coeff = m_dx.ptr()[dst_index];
        return {m_ret, m_ret + 2};
    }
};
}  // anonymous namespace

TEST_CASE("Symbolic.ANMImplicitSolver") {
    constexpr size_t batch = 5;
    TensorRNG rng;
    TensorND x0 = rng({batch}, 1, 2), dx = rng(x0.shape(), -2, -1);
    auto input_trans = std::make_shared<ANMImplicitSolverInputTrans>(dx);
    ComputingGraph cg;
    auto x = placeholder(cg), y = x.pow(1.5);
    ANMImplicitSolver solver{y.node(), input_trans,
                             SparseLinearDesc::make_identity(x0.shape()), x0,
                             0};
    printf("implicit:");
    fflush(stdout);
    int iter = 0;
    for (;;) {
        ++iter;
        REQUIRE(iter < 20);
        printf(" %.2g", solver.get_t_upper());
        fflush(stdout);
        if (solver.get_t_upper() >= 1) {
            break;
        }
        solver.update_approx();
    }
    printf(" (%d)\n", iter);

    TensorND xt;
    fp_t t;
    std::tie(xt, t) = solver.eval(solver.solve_a(1));
    REQUIRE(t == Approx(1));
    require_tensor_eq((xt + dx).pow(1.5), x0.pow(1.5));
}

TEST_CASE("Symbolic.PaperGeoExample") {
    TensorND coord_init{TensorShape{1, 2}}, df{coord_init.shape()};
    coord_init.woptr()[0] = 0;
    coord_init.woptr()[1] = -1;
    df.woptr()[0] = 0;
    df.woptr()[1] = -6;
    ComputingGraph cg;
    auto coord = placeholder(cg), x = coord.slice(1, 0, 1),
         y = coord.slice(1, 1, 2),
         f0 = linear_combine(
                 {{2, x.pow(2)}, {-5, x}, {1, y.pow(2)}, {-4, y}, {-2, x * y}},
                 -5),
         f1 = (x + 1).pow(2) + y.pow(2) - 2,
         f_all = concat(std::array<SymbolVar, 2>{f0, f1}, 1);
    auto iot_id = SparseLinearDesc::make_identity(coord_init.shape());
    ANMEqnSolver::HyperParam param;
    param.order = 20;
    ANMSolverVecScale anm{f_all.node(), iot_id, iot_id, coord_init, 0,
                          df,           param};

    FILE* fout = fopen("paper_geo_example.txt", "w");
    REQUIRE(fout);
    SANM_DEFER(std::bind(::fclose, fout));

    for (;;) {
        auto&& xtc = anm.xt_coeffs();
        sanm_assert(xtc[0].shape().total_nr_elems() == 3);
        for (auto&& i : xtc) {
            auto p = i.ptr();
            fprintf(fout, "%g %g %g ", p[0], p[1], p[2]);
        }
        fprintf(fout, "\n%g\n", anm.get_t_upper());
        if (anm.get_t_upper() >= 1) {
            break;
        }
        anm.update_approx();
    }
    auto print_err = [f_all](const TensorND& coordv) {
        auto errv = symbolic::eval_unary_func(f_all.node(), coordv);
        auto errp = errv.ptr();
        fp_t e0 = std::fabs(errp[0]), e1 = std::fabs(errp[1] - 6),
             erms = std::sqrt((e0 * e0 + e1 * e1) / 2);
        printf("err: %g %g; rms=%g\n", e0, e1, erms);
    };
    auto sol = anm.eval(anm.solve_a(1)).first;
    fprintf(fout, "%g %g\n", sol.ptr()[0], sol.ptr()[1]);
    print_err(sol);
    print_err(
            anm_general_solve("geoeg-eqn", f_all, coord_init, -df, 20, param));
}
