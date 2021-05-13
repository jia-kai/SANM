/**
 * \file fea/mesh.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

// utilities for general mesh support
#pragma once

#include "fea/material.h"
#include "fea/typedefs.h"
#include "libsanm/anm.h"

#include <Eigen/Core>

#include <span>
#include <vector>

namespace fea {

//! mesh vertex coordinates
template <int dim>
using CoordMatND = Eigen::Matrix<fp_t, dim, Eigen::Dynamic>;

//! mesh vertex coordinate masks
template <int dim>
using CoordMaskND = Eigen::Matrix<bool, dim, Eigen::Dynamic>;

//! vertex indicies in each element
template <int nvtx>
using IndexMat = Eigen::Matrix<int, nvtx, Eigen::Dynamic>;

/*!
 * \brief Replace values in \p dst from values in \p src when \p fixed_mask is
 *      false
 *
 * \param src_size number of elements in \p src. It is only used to check that
 *      all values in \p src are used. Setting this param to zero disables this
 *      check.
 */
void replace_with_mask(fp_t* dst, const bool* fixed_mask, const fp_t* src,
                       size_t dst_size, size_t src_size = 0);

template <int dim>
static inline void replace_with_mask(CoordMatND<dim>& dst,
                                     const CoordMaskND<dim>& mask,
                                     const fp_t* src, size_t src_size) {
    sanm_assert(dst.rows() == mask.rows() && dst.cols() == mask.cols());
    replace_with_mask(dst.data(), mask.data(), src, dst.rows() * dst.cols(),
                      src_size);
}

//! query which mesh elements contain a given vertex
class MeshVertexReverseList {
public:
    struct Item {
        leastsize_t ele;     //!< element index
        leastsize_t vtx_id;  //!< index of the vertex in the element

        Item() = default;
        Item(size_t ele_, size_t vtx_id_) : ele(ele_), vtx_id(vtx_id_) {}
    };

    using ItemSpan = std::span<const Item>;

    ItemSpan query(size_t vtx) const;

    template <int msize>
    static MeshVertexReverseList from_mesh(size_t nr_vtx,
                                           const IndexMat<msize>& mesh);

private:
    std::vector<Item> m_results;
    std::vector<std::pair<leastsize_t, leastsize_t>> m_vtx2loc;
};

//! transforming unknown vertex coordinates to the shape matrix to be futher
//! used for computing the deformation gradient
template <int dim, class Mesh>
class MeshShapeMatTrans final : public sanm::SparseLinearDescCompressed {
public:
    //! locating a cooridnate of a vertex
    struct VertexLoc {
        leastsize_t vtx;    //!<  ID of the vertex
        leastsize_t coord;  //!< coordinate number (0, 1, 2 for x, y, z)
    };
    using MeshPtr = std::shared_ptr<Mesh>;

    //! see DeformableBody::make_forward
    MeshShapeMatTrans(MeshPtr mesh, const CoordMaskND<dim>& fixed_mask,
                      const CoordMatND<dim>* init_vtx_coord,
                      const CoordMatND<dim>* vtx_delta);

    //! map from ID of the unknown to the corresponding vertex
    const std::vector<VertexLoc>& vertex_loc() const { return m_vertex_loc; }

    //! number of unknowns of the vertex coordinates
    size_t nr_unknown_vtx() const { return m_vertex_loc.size(); }

    sanm::TensorShape out_shape() const override {
        return {m_mesh->nr_faces(), dim, dim};
    }

    sanm::TensorShape inp_shape() const override { return {nr_unknown()}; }

    //! underlying tetrahedral mesh
    const Mesh& mesh() const { return *m_mesh; }

    //! the desired Ds/Dm matrix must be offseted by this bias
    const sanm::TensorND& bias() const { return m_bias; }

    //! initial value of the unknowns (i.e., the values in the current mesh)
    const sanm::TensorND& x0() const { return m_x0; }

    //! copy values associated with vertices (such as external force) to a
    //! flattened tensor
    sanm::TensorND copy_vtx_values(const CoordMatND<dim>& vtx_values) const;

private:
    const bool m_has_delta;
    MeshPtr m_mesh;
    sanm::TensorND m_bias, m_x0;
    std::vector<VertexLoc> m_vertex_loc;

    //! number of unknown values
    size_t nr_unknown() const { return m_vertex_loc.size() + m_has_delta; }
};

//! transforming the stress tensor to the force on the vertices by multiplying
//! with vertex norms
template <int dim, class Mesh>
class MeshForceOutputTrans final : public sanm::SparseLinearDescCompressed {
public:
    using InputTrans = MeshShapeMatTrans<dim, Mesh>;
    explicit MeshForceOutputTrans(std::shared_ptr<InputTrans> input_trans);

    sanm::TensorShape out_shape() const override {
        return {m_input_trans->nr_unknown_vtx()};
    }

    sanm::TensorShape inp_shape() const override {
        return m_input_trans->out_shape();
    }

private:
    std::shared_ptr<InputTrans> m_input_trans;
};

//! representing a deformable body
template <int dim, class Mesh>
class DeformableBody final : public sanm::NonCopyable {
public:
    //! a struct that describes internal elastic force of a deformable body
    struct ElasticForceModel {
        sanm::symbolic::ComputingGraph cg;
        sanm::SymbolVar y;  //!< stress tensor, fed to #lt_out for node force
        sanm::SymbolVar potential;  //!< potential energy; may be null
        std::shared_ptr<MeshShapeMatTrans<dim, Mesh>> lt_inp;
        std::shared_ptr<MeshForceOutputTrans<dim, Mesh>> lt_out;
    };
    using MeshPtr = std::shared_ptr<Mesh>;

    DeformableBody(const MaterialProperty& material, MeshPtr mesh);

    //! the underlying material
    const MaterialProperty& material() const { return m_material; }

    //! the underlying tetrahedron mesh
    const Mesh& mesh() const { return *m_mesh; }

    //! mutable mask of the vertex coordinates; true values are those that are
    //! fixed (i.e., should not be treated as an unknown)
    CoordMaskND<dim>& coord_fixed_mask() { return m_coord_fixed_mask; }
    const CoordMaskND<dim>& coord_fixed_mask() const {
        return m_coord_fixed_mask;
    }

    /*!
     * \brief make an elastic shape model for solving the rest shape given the
     *      deformed shape (which is this one)
     *
     * Note: this TetrahedralDeformableBody can be safely destructed after the
     * model is created.
     */
    std::unique_ptr<ElasticForceModel> make_inverse(
            EnergyModel energy_model) const;

    /*!
     * \brief make an elastic shape model for solving the deformed shape given
     *      the rest shape (which is this one)
     *
     * \param init_vtx_coord initial vertex location; if not provided, it is
     *      assumed to be the rest shape of the mesh
     *
     * \param vtx_delta the delta of fixed coordinates for implicit
     *      continuation. The system will have one more unknown t, and the x
     *      coordinate is x0 + t * vtx_delta
     *
     * Note: this TetrahedralDeformableBody can be safely destructed after the
     * model is created.
     */
    std::unique_ptr<ElasticForceModel> make_forward(
            EnergyModel energy_model,
            const CoordMatND<dim>* init_vtx_coord = nullptr,
            const CoordMatND<dim>* vtx_delta = nullptr) const;

    /*!
     * \brief check whether the solution to an elastic model meets certain
     *      specifications
     * \param xt the solution
     * \param f_load the load force corresponding to \p xt
     * \param final_mesh the deformed mesh after replacing verticies with values
     *      from \p xt
     * \return force RMS
     */
    static fp_t solution_sanity_check(const ElasticForceModel& model,
                                      const sanm::TensorND& xt,
                                      const sanm::TensorND& f_load,
                                      const Mesh& final_mesh);

private:
    MaterialProperty m_material;
    MeshPtr m_mesh;

    CoordMaskND<dim> m_coord_fixed_mask;
};
}  // namespace fea
