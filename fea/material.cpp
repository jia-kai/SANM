/**
 * \file fea/material.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "fea/material.h"

using namespace fea;

MaterialProperty MaterialProperty::from_young_poisson(fp_t E, fp_t nu) {
    MaterialProperty ret;
    ret.m_young_modulus = E;
    ret.m_poisson_ratio = nu;
    ret.m_bulk_modulus = E / (3 * (1 - nu * 2));
    ret.m_shear_modulus = E / (2 * (1 + nu));
    ret.m_lame_first = E * nu / ((1 + nu) * (1 - nu * 2));
    return ret;
}

sanm::SymbolVar fea::cauchy_stress(EnergyModel energy_model,
                                   const MaterialProperty& material,
                                   sanm::SymbolVar F, size_t dim) {
    using namespace sanm;
    using namespace sanm::symbolic;
    switch (energy_model) {
        case EnergyModel::NEOHOOKEAN_I: {
            fp_t k = material.bulk_modulus(), mu = material.shear_modulus();
            SymbolVar b = F.batched_matmul(F.batched_transpose()),
                      J = F.batched_det(), Ic = F.pow(2).reduce_sum(-1),
                      J53 = J.pow(-5. / 3.),
                      t2 = linear_combine({{mu / (-3._fp), J53 * Ic}, {k, J}},
                                          -k)
                                   .batched_mul_eye(dim),
                      sigma = linear_combine({{mu, J53 * b}, {1._fp, t2}});
            return sigma;
        }
        case EnergyModel::NEOHOOKEAN_C: {
            fp_t lambda = material.lame_first(), mu = material.shear_modulus();
            SymbolVar b = F.batched_matmul(F.batched_transpose()),
                      Jinv = F.batched_det().pow(-1),
                      xI = linear_combine(
                              {{mu, Jinv}, {lambda, Jinv * Jinv.log()}}),
                      sigma = linear_combine(
                              {{mu, Jinv * b},
                               {-1._fp, xI.batched_mul_eye(dim)}});
            return sigma;
        }
        default:
            throw SANMError{
                    ssprintf("cauchy_stress unimplemented for energy model %d",
                             static_cast<int>(energy_model))};
    }
}

sanm::SymbolVar fea::pk1(EnergyModel energy_model,
                         const MaterialProperty& material, sanm::SymbolVar F,
                         size_t dim) {
    using namespace sanm;
    using namespace sanm::symbolic;
    switch (energy_model) {
        case EnergyModel::NEOHOOKEAN_I: {
            fp_t k = material.bulk_modulus(), mu = material.shear_modulus();
            SymbolVar FTinv = batched_mat_inv_mul(F, {}, true)
                                      .batched_transpose(),
                      J = F.batched_det(), Ic = F.pow(2).reduce_sum(-1),
                      J23 = J.pow(-2. / 3.),
                      t2 = linear_combine({{mu / (-3._fp), J23 * Ic},
                                           {k, J * J},
                                           {-k, J}},
                                          0) *
                           FTinv,
                      P = linear_combine({{mu, J23 * F}, {1._fp, t2}});
            return P;
        }
        case EnergyModel::NEOHOOKEAN_C: {
            fp_t mu = material.shear_modulus(), lambda = material.lame_first();
            SymbolVar FTinv = batched_mat_inv_mul(F, {}, true)
                                      .batched_transpose(),
                      J = F.batched_det(),
                      P = linear_combine({{mu, F},
                                          {-mu, FTinv},
                                          {lambda, J.log() * FTinv}});
            return P;
        }
        case EnergyModel::ARAP: {
            fp_t mu = material.shear_modulus();
            return (F - F.batched_svd_w(true)[2]) * mu;
        }
        case EnergyModel::StVK_STRETCH: {
            fp_t mu = material.shear_modulus();
            return linear_combine({{mu, F.batched_matmul(F.batched_transpose())
                                                .batched_matmul(F)},
                                   {-mu, F}});
        }
        default:
            throw SANMError{ssprintf("pk1 unimplemented for energy model %d",
                                     static_cast<int>(energy_model))};
    }
}

sanm::SymbolVar fea::elastic_potential_density(EnergyModel energy_model,
                                               const MaterialProperty& material,
                                               sanm::SymbolVar F, size_t dim) {
    using namespace sanm;
    using namespace sanm::symbolic;
    switch (energy_model) {
        case EnergyModel::ARAP: {
            fp_t mu = material.shear_modulus();
            return (F - F.batched_svd_w(true)[2]).pow(2).reduce_sum(-1) *
                   (mu / 2);
        }
        default:
            return {};
    }
}
