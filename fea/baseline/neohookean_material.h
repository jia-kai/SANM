#pragma once

#include "material.h"

namespace materials {

template <int dim, typename T>
class CompressibleNeohookeanMaterial : public Material<dim, T> {
    using Super = Material<dim, T>;

public:
    using typename Super::MatrixDim2T;
    using typename Super::MatrixDimT;

    CompressibleNeohookeanMaterial(const T young_modulus, const T poisson_ratio)
            : Super(young_modulus, poisson_ratio) {}

    CompressibleNeohookeanMaterial(
            const CompressibleNeohookeanMaterial<dim, T>& material)
            : Super(material) {}

    T EnergyDensity(const MatrixDimT& F) const override;

    MatrixDimT StressTensor(const MatrixDimT& F) const override;

    MatrixDimT StressDifferential(const MatrixDimT& F,
                                  const MatrixDimT& dF) const override;

    MatrixDim2T StressDifferential(const MatrixDimT& F) const override;
};

template <int dim, typename T>
class IncompressibleNeohookeanMaterial : public Material<dim, T> {
    using Super = Material<dim, T>;

public:
    using typename Super::MatrixDim2T;
    using typename Super::MatrixDimT;

    IncompressibleNeohookeanMaterial(const T young_modulus,
                                     const T poisson_ratio)
            : Super(young_modulus, poisson_ratio) {}

    IncompressibleNeohookeanMaterial(
            const CompressibleNeohookeanMaterial<dim, T>& material)
            : Super(material) {}

    T EnergyDensity(const MatrixDimT& F) const override;

    MatrixDimT StressTensor(const MatrixDimT& F) const override;

    MatrixDimT StressDifferential(const MatrixDimT& F,
                                  const MatrixDimT& dF) const override;

    MatrixDim2T StressDifferential(const MatrixDimT& F) const override;
};

}  // namespace materials
