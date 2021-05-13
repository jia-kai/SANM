#include "arap_material.h"
#include "main.h"
#include "utils.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace materials;

namespace {
template <int dim, typename T>
void svd_w(const Eigen::Matrix<T, dim, dim>& mat_inp,
           Eigen::Matrix<T, dim, dim>& U, Eigen::Matrix<T, dim, 1>& S,
           Eigen::Matrix<T, dim, dim>& V) {
    using Mat = Eigen::Matrix<T, dim, dim>;
    Eigen::JacobiSVD<Mat> svd{mat_inp,
                              Eigen::ComputeFullU | Eigen::ComputeFullV};
    S = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();
    if ((U.determinant() < 0) != (V.determinant() < 0)) {
#if 0
        // code copied from libsan SVDW to maintain consistency
        constexpr double EPS = 1e-3;
        int best_idx = -1, best_idx_nr = dim + 1;
        for (size_t i = 0; i < dim; ++i) {
            size_t j = i + 1;
            // ms already sorted
            while (j < dim && std::fabs(S(i) - S(j)) < EPS) {
                ++j;
            }
            int nr = j - i;
            // best case is to negate an odd number of smallest singular
            // values (so si+sj != 0 in the hessian);
            // otherwise negate one value whose has the least
            // repetitionss
            if (nr <= best_idx_nr || (nr == best_idx_nr + 1 && nr % 2 == 1)) {
                best_idx = i;
                best_idx_nr = nr;
                if (nr == 1) {
                    break;
                }
            }
            i = j;
        }
        if (best_idx_nr == 1 || best_idx_nr % 2 == 0) {
            U.col(best_idx) = -U.col(best_idx);
            S(best_idx) = -S(best_idx);
        } else {
            for (int i = best_idx; i < best_idx + best_idx_nr; ++i) {
                U.col(i) = -U.col(i);
                S(i) = -S(i);
            }
        }
#else
        U.col(dim - 1) = -U.col(dim - 1);
        S(dim - 1) = -S(dim - 1);
#endif
    }
}
}  // namespace

template <int dim, typename T>
T ARAPElasticityMaterial<dim, T>::EnergyDensity(const MatrixDimT& F) const {
    MatrixDimT U, V, R;
    Eigen::Matrix<T, dim, 1> S;
    svd_w(F, U, S, V);
    R.noalias() = U * V.transpose();
    return (F - R).squaredNorm() * (this->mu() * 0.5);
}

template <int dim, typename T>
typename ARAPElasticityMaterial<dim, T>::MatrixDimT
ARAPElasticityMaterial<dim, T>::StressTensor(const MatrixDimT& F) const {
    MatrixDimT U, V, R;
    Eigen::Matrix<T, dim, 1> S;
    svd_w(F, U, S, V);
    R.noalias() = U * V.transpose();
    return (F - R) * this->mu();
}

template <int dim, typename T>
typename ARAPElasticityMaterial<dim, T>::MatrixDimT
ARAPElasticityMaterial<dim, T>::StressDifferential(const MatrixDimT& F,
                                                   const MatrixDimT& dF) const {
    throw std::runtime_error{"unimplemented"};
}

template <int dim, typename T>
typename ARAPElasticityMaterial<dim, T>::MatrixDim2T
ARAPElasticityMaterial<dim, T>::StressDifferential(const MatrixDimT& F) const {
    cf_assert(dim == 3);
    MatrixDimT U, V;
    Eigen::Matrix<T, dim, 1> S;
    svd_w(F, U, S, V);
    Eigen::Matrix<T, 3, 3> T0, T1, T2;
    T0 << 0, -1, 0, 1, 0, 0, 0, 0, 0;
    T0 = std::sqrt(0.5) * U * T0 * V.transpose();
    T1 << 0, 0, 0, 0, 0, 1, 0, -1, 0;
    T1 = std::sqrt(0.5) * U * T1 * V.transpose();
    T2 << 0, 0, 1, 0, 0, 0, -1, 0, 0;
    T2 = std::sqrt(0.5) * U * T2 * V.transpose();

    Eigen::Map<Eigen::Matrix<T, 9, 1>> t0{T0.data()}, t1{T1.data()},
            t2{T2.data()};
    T s0 = S(0), s1 = S(1), s2 = S(2);
    Eigen::Matrix<T, 9, 9> H;
    H.setIdentity();
    T (*clip)(T);
    if (baseline::g_hessian_proj) {
        clip = [](T x) { return std::max<T>(x, 2); };
    } else {
        clip = [](T x) { return x; };
    }
    H.noalias() -= 2 / (clip(s0 + s1)) * t0 * t0.transpose();
    H.noalias() -= 2 / (clip(s1 + s2)) * t1 * t1.transpose();
    H.noalias() -= 2 / (clip(s0 + s2)) * t2 * t2.transpose();
    return H * this->mu();
}

template class materials::ARAPElasticityMaterial<3, double>;
