#pragma once

#include "material.h"

namespace materials {

//! as-rigid-as-possible
template <int dim, typename T>
class ARAPElasticityMaterial : public Material<dim, T> {
    using Super = Material<dim, T>;

public:
    using typename Super::MatrixDim2T;
    using typename Super::MatrixDimT;

    ARAPElasticityMaterial(const T young_modulus, const T poisson_ratio)
            : Super(young_modulus, poisson_ratio) {}

    ARAPElasticityMaterial(const ARAPElasticityMaterial<dim, T>& material)
            : Super(material) {}

    T EnergyDensity(const MatrixDimT& F) const override;

    MatrixDimT StressTensor(const MatrixDimT& F) const override;

    MatrixDimT StressDifferential(const MatrixDimT& F,
                                  const MatrixDimT& dF) const override;

    MatrixDim2T StressDifferential(const MatrixDimT& F) const override;
};

}  // namespace materials
