/**
 * \file fea/material.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// material models
#pragma once

#include "fea/typedefs.h"
#include "libsanm/oprs.h"

namespace fea {
/*!
 * \brief Descriptor for material elastic deformation properties
 *
 * See https://en.wikipedia.org/wiki/Elastic_modulus for a complete conversion
 * table.
 */
class MaterialProperty final {
    fp_t m_young_modulus = 0, m_poisson_ratio = 0, m_bulk_modulus = 0,
         m_shear_modulus = 0, m_lame_first = 0, m_density = 0;

public:
    //! properties from Young's modulus (E) and Poisson's ratio (\f$\nu\f$)
    static MaterialProperty from_young_poisson(fp_t E, fp_t nu);

    //! Bulk modulus (K or \f$\kappa\f$)
    fp_t bulk_modulus() const { return m_bulk_modulus; }

    //! Young's modulus (E)
    fp_t young_modulus() const { return m_young_modulus; }

    //! Shear modulus (G), a.k.a. Lamé's second parameter (\f$\mu\f$)
    fp_t shear_modulus() const { return m_shear_modulus; }

    //! Poisson's ratio (\f$\nu\f$)
    fp_t poisson_ratio() const { return m_poisson_ratio; }

    //! Lamé's first parameter (\f$\lambda\f$)
    fp_t lame_first() const { return m_lame_first; }

    fp_t density() const { return m_density; }

    MaterialProperty& set_density(fp_t density) {
        m_density = density;
        return *this;
    }
};

enum class EnergyModel {
    NEOHOOKEAN_I,  //!< incompressible neo-hookean
    NEOHOOKEAN_C,  //!< compressible neo-hookean
    ARAP,          //!< as-rigid-as-possible
    StVK_STRETCH,  //!< the stretch term in St. Venant Kirchhoff
};

/*!
 * \brief Compute the symbolic Cauchy stress tensor from given deformation
 *      gradient
 *
 * Note that the Cauchy stress tensor relates the traction vector linearly to
 * the norms in the deformed state.
 *
 * \param F the deformation gradient defined as dx/dX
 * \param dim the dimension of the space
 */
sanm::SymbolVar cauchy_stress(EnergyModel energy_model,
                              const MaterialProperty& material,
                              sanm::SymbolVar F, size_t dim);

//! compute the first Piola-Kirchhoff stress tensor
//! see also cauchy_stress()
sanm::SymbolVar pk1(EnergyModel energy_model, const MaterialProperty& material,
                    sanm::SymbolVar F, size_t dim);

//! compute the elastic potential energy density for each element; return empty
//! (instead of throwing) if not supported
sanm::SymbolVar elastic_potential_density(EnergyModel energy_model,
                                          const MaterialProperty& material,
                                          sanm::SymbolVar F, size_t dim);
}  // namespace fea
