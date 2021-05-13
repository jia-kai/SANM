#include "neohookean_material.h"
#include "main.h"
#include "utils.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace materials;

template <int dim, typename T>
T CompressibleNeohookeanMaterial<dim, T>::EnergyDensity(
        const MatrixDimT& F) const {
    const T I1 = (F.transpose() * F).trace();
    const T J = F.determinant();
    if (J <= 0) {
        throw NumericalError{"J <= 0 in CompressibleNeohookeanMaterial"};
    }
    const T log_J = log(J);
    const T mu = Super::mu();
    const T lambda = Super::lambda();
    return mu / 2.0 * (I1 - dim) - mu * log_J + lambda / 2.0 * log_J * log_J;
}

template <int dim, typename T>
typename CompressibleNeohookeanMaterial<dim, T>::MatrixDimT
CompressibleNeohookeanMaterial<dim, T>::StressTensor(
        const MatrixDimT& F) const {
    const T J = F.determinant();
    if (J <= 0) {
        throw NumericalError{"J <= 0 in CompressibleNeohookeanMaterial"};
    }
    const MatrixDimT F_inv_trans = F.inverse().transpose();
    const T mu = Super::mu();
    const T lambda = Super::lambda();
    return mu * (F - F_inv_trans) + lambda * log(J) * F_inv_trans;
}

template <int dim, typename T>
typename CompressibleNeohookeanMaterial<dim, T>::MatrixDimT
CompressibleNeohookeanMaterial<dim, T>::StressDifferential(
        const MatrixDimT& F, const MatrixDimT& dF) const {
    throw std::runtime_error{"not implemented"};
}

template <int dim, typename Td>
typename CompressibleNeohookeanMaterial<dim, Td>::MatrixDim2T
CompressibleNeohookeanMaterial<dim, Td>::StressDifferential(
        const MatrixDimT& F) const {
    const Td mu = Super::mu();
    const Td lambda = Super::lambda();

    using Vec = Eigen::Matrix<Td, dim, 1>;
    using Vec2 = Eigen::Matrix<Td, dim * dim, 1>;
    Eigen::JacobiSVD<MatrixDimT> svd{F,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV};
    Vec S = svd.singularValues();
    MatrixDimT U = svd.matrixU(), V = svd.matrixV(), A;

    // construct the matrix A
    Td I3 = S(0) * S(1) * S(2), Adia = lambda * (1 - std::log(I3)) + mu;
    for (int i = 0; i < 3; ++i) {
        A(i, i) = Adia / (S(i) * S(i)) + mu;
        for (int j = i + 1; j < 3; ++j) {
            A(i, j) = A(j, i) = lambda / (S(i) * S(j));
        }
    }

    Vec2 eigs;
    Eigen::SelfAdjointEigenSolver<MatrixDimT> eigensolver{A};
    cf_assert(eigensolver.info() == Eigen::Success);
    Eigen::Map<Vec>{eigs.data()} = eigensolver.eigenvalues();

    {
        Td eig6tmp = lambda * log(I3) - mu;
#define SET(t, i, j) eigs[t + 3] = eig6tmp / (S(i) * S(j)) + mu
        SET(0, 0, 1);
        SET(1, 1, 2);
        SET(2, 0, 2);
        eig6tmp = -eig6tmp;
        SET(3, 0, 1);
        SET(4, 1, 2);
        SET(5, 0, 2);
#undef SET
    }

    MatrixDimT D0, D1, D2, T[6];
    D0.noalias() = U.col(0) * V.col(0).transpose();
    D1.noalias() = U.col(1) * V.col(1).transpose();
    D2.noalias() = U.col(2) * V.col(2).transpose();
    auto compute_Q_first3 = [&](MatrixDimT& Q, int i) {
        auto& evec = eigensolver.eigenvectors();
        Td z0 = evec(0, i), z1 = evec(1, i), z2 = evec(2, i);
        Q = z0 * D0 + z1 * D1 + z2 * D2;
    };
    T[0] << 0, -1, 0, 1, 0, 0, 0, 0, 0;
    T[1] << 0, 0, 0, 0, 0, 1, 0, -1, 0;
    T[2] << 0, 0, 1, 0, 0, 0, -1, 0, 0;
    T[3] << 0, 1, 0, 1, 0, 0, 0, 0, 0;
    T[4] << 0, 0, 0, 0, 0, 1, 0, 1, 0;
    T[5] << 0, 0, 1, 0, 0, 0, 1, 0, 0;

    MatrixDimT Q;
    MatrixDim2T H;
    H.setZero();
    int nr_pos = 0;
    for (int i = 0; i < 9; ++i) {
        if (baseline::g_hessian_proj && eigs[i] < 0) {
            continue;
        }
        ++nr_pos;
        if (i < 3) {
            compute_Q_first3(Q, i);
        } else {
            Q.noalias() = U * T[i - 3] * V.transpose() * std::sqrt(0.5);
        }
        Eigen::Map<Vec2> qf{Q.data()};
        H.noalias() += qf * qf.transpose() * eigs[i];
    }
    cf_assert(nr_pos);
    return H;
}

template <int dim, typename T>
T IncompressibleNeohookeanMaterial<dim, T>::EnergyDensity(
        const MatrixDimT& F) const {
    const T Ic = (F.transpose() * F).trace();
    const T J = F.determinant();
    if (J <= 0) {
        throw NumericalError{"J <= 0 in IncompressibleNeohookeanMaterial"};
    }
    const T mu = Super::mu();
    const T k = Super::bulk_modulus();
    return mu / T(2) * (std::pow(J, T(-2.0 / 3.0)) * Ic - 3) +
           k / T(2) * (J - 1) * (J - 1);
}

template <int dim, typename T>
typename IncompressibleNeohookeanMaterial<dim, T>::MatrixDimT
IncompressibleNeohookeanMaterial<dim, T>::StressTensor(
        const MatrixDimT& F) const {
    const T J = F.determinant();
    if (J <= 0) {
        throw NumericalError{"J <= 0 in IncompressibleNeohookeanMaterial"};
    }
    const T Ic = (F.transpose() * F).trace();
    const MatrixDimT F_inv_trans = F.inverse().transpose();
    const T mu = Super::mu();
    const T k = Super::bulk_modulus();
    return (F - F_inv_trans * (Ic / T(3))) * (mu * std::pow(J, T(-2.0 / 3.0))) +
           F_inv_trans * (k * J * (J - 1));
}

template <int dim, typename T>
typename IncompressibleNeohookeanMaterial<dim, T>::MatrixDimT
IncompressibleNeohookeanMaterial<dim, T>::StressDifferential(
        const MatrixDimT& F, const MatrixDimT& dF) const {
    throw std::runtime_error{"not implemented"};
}

template <int dim, typename Td>
typename IncompressibleNeohookeanMaterial<dim, Td>::MatrixDim2T
IncompressibleNeohookeanMaterial<dim, Td>::StressDifferential(
        const MatrixDimT& F) const {
    const Td mu = Super::mu();
    const Td k = Super::bulk_modulus();

    using Vec = Eigen::Matrix<Td, dim, 1>;
    using Vec2 = Eigen::Matrix<Td, dim * dim, 1>;
    Eigen::JacobiSVD<MatrixDimT> svd{F,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV};
    Vec S = svd.singularValues();
    MatrixDimT U = svd.matrixU(), V = svd.matrixV(), A;

    // construct the matrix A
    auto sqr = [](Td x) { return x * x; };
    Td I3 = S(0) * S(1) * S(2), I3_83 = std::pow(I3, Td(8.0 / 3)),
       I3_53 = std::pow(I3, Td(5.0 / 3)),
       S2[3] = {sqr(S(0)), sqr(S(1)), sqr(S(2))}, I2 = S2[0] + S2[1] + S2[2];
    for (int i = 0; i < 3; ++i) {
        int o0 = i == 0 ? 1 : 0, o1 = i == 2 ? 1 : 2;
        A(i, i) = S2[o0] * S2[o1] *
                  (9 * I3_83 * k + mu * (2 * S2[i] + 5 * (S2[o0] + S2[o1]))) /
                  (9 * I3_83);
        for (int j = i + 1; j < 3; ++j) {
            A(i, j) = A(j, i) =
                    S(3 - i - j) *
                    (9 * k * (I3_83 * 2 - I3_53) +
                     mu * 2 * (S2[3 - i - j] - 2 * S2[i] - 2 * S2[j])) /
                    (9 * I3_53);
        }
    }

    Vec2 eigs;
    Eigen::SelfAdjointEigenSolver<MatrixDimT> eigensolver{A};
    cf_assert(eigensolver.info() == Eigen::Success);
    Eigen::Map<Vec>{eigs.data()} = eigensolver.eigenvalues();

    {
        Td eig6tmp = I3_53 * 3 * k * (I3 - 1) - I2 * mu;
#define SET(t, i, j) \
    eigs[t + 3] = (eig6tmp + 3 * mu * S(i) * S(j)) * S(3 - i - j) / (3 * I3_53)
        SET(0, 0, 1);
        SET(1, 1, 2);
        SET(2, 0, 2);
        eig6tmp = -eig6tmp;
        SET(3, 0, 1);
        SET(4, 1, 2);
        SET(5, 0, 2);
#undef SET
    }

    MatrixDimT D0, D1, D2, T[6];
    D0.noalias() = U.col(0) * V.col(0).transpose();
    D1.noalias() = U.col(1) * V.col(1).transpose();
    D2.noalias() = U.col(2) * V.col(2).transpose();
    auto compute_Q_first3 = [&](MatrixDimT& Q, int i) {
        auto& evec = eigensolver.eigenvectors();
        Td z0 = evec(0, i), z1 = evec(1, i), z2 = evec(2, i);
        Q = z0 * D0 + z1 * D1 + z2 * D2;
    };
    T[0] << 0, -1, 0, 1, 0, 0, 0, 0, 0;
    T[1] << 0, 0, 0, 0, 0, 1, 0, -1, 0;
    T[2] << 0, 0, 1, 0, 0, 0, -1, 0, 0;
    T[3] << 0, 1, 0, 1, 0, 0, 0, 0, 0;
    T[4] << 0, 0, 0, 0, 0, 1, 0, 1, 0;
    T[5] << 0, 0, 1, 0, 0, 0, 1, 0, 0;

    MatrixDimT Q;
    MatrixDim2T H;
    H.setZero();
    int nr_pos = 0;
    for (int i = 0; i < 9; ++i) {
        if (baseline::g_hessian_proj && eigs[i] < 0) {
            continue;
        }
        ++nr_pos;
        if (i < 3) {
            compute_Q_first3(Q, i);
        } else {
            Q.noalias() = U * T[i - 3] * V.transpose() * std::sqrt(0.5);
        }
        Eigen::Map<Vec2> qf{Q.data()};
        H.noalias() += qf * qf.transpose() * eigs[i];
    }
    cf_assert(nr_pos);
    return H;
}

template class materials::CompressibleNeohookeanMaterial<3, double>;
template class materials::IncompressibleNeohookeanMaterial<3, double>;
