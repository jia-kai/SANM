#pragma once
#include <Eigen/Core>

namespace materials{

    template <typename T>
    using Matrix3X = Eigen::Matrix<T, 3, Eigen::Dynamic>;

    template <typename T>
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    using Matrix4Xi = Eigen::Matrix<int, 4, Eigen::Dynamic>;

    template <typename T>
    using Matrix8Xi = Eigen::Matrix<int, 8, Eigen::Dynamic>;

    template <typename T>
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
}
