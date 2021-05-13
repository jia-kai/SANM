/**
 * \file tests/ano.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/ano.h"
#include "libsanm/oprs.h"
#include "tests/helper.h"

using namespace sanm;
using namespace symbolic;

namespace {
using CoeffSolver = ANOMinimizer::CoeffSolver;
void run_minimize(const char* name, ANOMinimizer& amin, int maxiter = 20,
                  fp_t grad_norm = 1e-3) {
    printf("min %s losses:", name);
    fflush(stdout);
    int iter = 0;
    fp_t cur_grad_norm;
    while ((cur_grad_norm = amin.grad_l2()) > grad_norm && iter < maxiter) {
        ++iter;
        printf(" %.3f", amin.loss());
        fflush(stdout);
        auto stat = amin.update_approx();
        printf("(%.3f/%.3f,%.3f)", stat.a_m, stat.a_bound, cur_grad_norm);
    }
    printf(" iter=%d loss=%g\n", iter, amin.loss());
}

SymbolVar sum3(SymbolVar x, SymbolVar y, fp_t bias) {
    return linear_combine({{1.f, x}, {1.f, y}}, bias);
}

TensorND scalar(fp_t v) {
    TensorND ret{TensorShape{1}};
    ret.woptr()[0] = v;
    return ret;
}

fp_t test_rosenbrock(unary_polynomial::coeff_t x0, int maxiter) {
    // test on the Rosenbrock function
    // we use a naive expansion because tensor indexing opr has not been
    // implemented
    ComputingGraph cg;
    std::vector<SymbolVar> xs(x0.size());
    TensorValueMap x0_map;
    for (size_t i = 0; i < x0.size(); ++i) {
        xs[i] = placeholder(cg);
        x0_map.insert(xs[i].node(), scalar(x0[i]));
    }

    std::vector<std::pair<fp_t, SymbolVar>> sum_desc;
    for (size_t i = 0; i + 1 < x0.size(); ++i) {
        sum_desc.push_back({100._fp, (xs[i + 1] - xs[i].pow(2)).pow(2)});
        sum_desc.push_back({1._fp, (1 - xs[i]).pow(2)});
    }

    SymbolVar loss = linear_combine(sum_desc, 0);
    ANOMinimizer amin{loss.node(), x0_map, CoeffSolver::make_gd_approx(0.1)};
    run_minimize(ssprintf("Rosenbrock(%zu)", x0.size()).c_str(), amin, maxiter);
    return amin.loss();
}

}  // anonymous namespace

TEST_CASE("ANO.SimpleTwo") {
    // minimize a two-var function:
    // ((y-4)^2*(x-0.5)^3 + 2)^0.3 + ln((x-2)^2 + (x+y-3)^2 + 1)
    ComputingGraph cg;
    auto x = placeholder(cg), y = placeholder(cg),
         loss = ((y - 4).pow(2) * (x - 0.5).pow(3) + 2).pow(0.3) +
                sum3((x - 2).pow(2), sum3(x, y, -3).pow(2), 1).log();
    TensorValueMap x0;
    x0.insert(x.node(), scalar(3));
    x0.insert(y.node(), scalar(3));

    ANOMinimizer amin{loss.node(), x0, CoeffSolver::make_gd_approx(0.4)};
    fp_t loss0 = amin.loss();
    run_minimize("simple-two", amin);
    auto sol = amin.get_x();
    printf("sol: x=%g y=%g\n", sol.get(x.node()).ptr()[0],
           sol.get(y.node()).ptr()[0]);
    REQUIRE(amin.loss() < loss0 / 2);
}

TEST_CASE("ANO.RosenbrockScipy") {
    // compare with scipy example
    // https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize
    // the convergence is much slower than second order methods, initially
    // faster than gradient descent with a fixed learning rate, but somehow it
    // falls into a local minimum
    std::vector<fp_t> x0 = {1.3, 0.7, 0.8, 1.9, 1.2};
    fp_t loss = test_rosenbrock(x0, 20);
    REQUIRE(loss < 0.5);
}
