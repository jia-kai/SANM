#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace materials {

//! error thrown when evaluation of material stress tensor fails
class NumericalError : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

template <int dim, typename T>
class Material {
public:
    using MatrixDimT = Eigen::Matrix<T, dim, dim>;
    using MatrixDim2T = Eigen::Matrix<T, dim * dim, dim * dim>;

    Material(const T young_modulus, const T poisson_ratio)
            : young_modulus_(young_modulus),
              poisson_ratio_(poisson_ratio),
              mu_(young_modulus / 2.0 / (1 + poisson_ratio)),
              lambda_(young_modulus * poisson_ratio / (1 + poisson_ratio) /
                      (1 - 2 * poisson_ratio)),
              bulk_modulus_{young_modulus / (3 * (1 - poisson_ratio * 2))} {}

    Material(const Material<dim, T>& material)
            : Material(material.young_modulus_, material.poisson_ratio_) {}

    virtual ~Material() = default;

    double young_modulus() const { return young_modulus_; }
    double poisson_ratio() const { return poisson_ratio_; }
    double mu() const { return mu_; }
    double lambda() const { return lambda_; }
    double bulk_modulus() const { return bulk_modulus_; }

    virtual T EnergyDensity(const MatrixDimT& F) const = 0;
    virtual MatrixDimT StressTensor(const MatrixDimT& F) const = 0;
    virtual MatrixDimT StressDifferential(const MatrixDimT& F,
                                          const MatrixDimT& dF) const = 0;

    virtual MatrixDim2T StressDifferential(const MatrixDimT& F) const = 0;

private:
    // Intentionally disable the copy assignment since all data members are
    // constant. Do not provide the definition so that we won't be able to
    // accidentally use it in member functions.
    Material<dim, T>& operator=(const Material<dim, T>&) = delete;

    const double young_modulus_;
    const double poisson_ratio_;
    const double mu_;
    const double lambda_;
    const double bulk_modulus_;
};

}  // namespace materials

