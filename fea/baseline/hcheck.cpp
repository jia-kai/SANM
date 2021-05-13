#include "hcheck.h"
#include "utils.h"

#include <iostream>

void materials::check_hessian(const char* name,
                              const Material<3, double>& material) {
    constexpr double eps = 1e-4;
    Eigen::Matrix<double, 3, 3> F, P0, P1, Pdiff;
    // a random matrix generated in numpy, with a positive determinant
    F << 0.8050644, -0.74382229, 0.16722774, 0.9854287, 0.8050644, 0.24858991,
            0.66782994, -0.6447535, 1.12759299;

    bool succ = true;
    Eigen::Matrix<double, 3, 3> Gnum, Gcomp = material.StressTensor(F);
    for (int fj = 0; fj < 3; ++fj) {
        for (int fi = 0; fi < 3; ++fi) {
            double v0 = F(fi, fj);
            F(fi, fj) = v0 + eps;
            double E0 = material.EnergyDensity(F);
            F(fi, fj) = v0 - eps;
            double E1 = material.EnergyDensity(F);
            F(fi, fj) = v0;
            Gnum(fi, fj) = (E0 - E1) / (eps * 2);
        }
    }
    printf("======== begin G,H check: %s\n", name);
    Eigen::IOFormat fmt(3, 0, ", ", ";\n", "", "", "[", "]");
    std::cout << Gnum.format(fmt) << std::endl;
    std::cout << Gcomp.format(fmt) << std::endl;
    double maxdiff = 0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            double h0 = Gnum(i, j), h1 = Gcomp(i, j);
            maxdiff = std::max(maxdiff, std::fabs(h0 - h1));
        }
    }
    if (maxdiff > 1e-5) {
        printf("!!!!!!!!!! maxdiff=%g\n", maxdiff);
        succ = false;
    } else {
        printf("maxdiff=%g\n", maxdiff);
    }

    Eigen::Matrix<double, 9, 9> Hnum, Hcomp = material.StressDifferential(F);
    for (int fj = 0; fj < 3; ++fj) {
        for (int fi = 0; fi < 3; ++fi) {
            double v0 = F(fi, fj);
            F(fi, fj) = v0 + eps;
            P0 = material.StressTensor(F);
            F(fi, fj) = v0 - eps;
            P1 = material.StressTensor(F);
            F(fi, fj) = v0;
            Pdiff = (P0 - P1) / (eps * 2);
            Hnum.col(fi + fj * 3) =
                    Eigen::Map<Eigen::Matrix<double, 9, 1>>{Pdiff.data()};
        }
    }

    std::cout << Hnum.format(fmt) << std::endl;
    std::cout << Hcomp.format(fmt) << std::endl;
    maxdiff = 0;
    for (int j = 0; j < 9; ++j) {
        for (int i = 0; i < 9; ++i) {
            double h0 = Hnum(i, j), h1 = Hcomp(i, j);
            maxdiff = std::max(maxdiff, std::fabs(h0 - h1));
        }
    }
    if (maxdiff > 1e-5) {
        printf("!!!!!!!!!! maxdiff=%g\n", maxdiff);
        succ = false;
    } else {
        printf("maxdiff=%g\n", maxdiff);
    }
    printf("======== end G,H check\n");
    cf_assert(succ);
}
