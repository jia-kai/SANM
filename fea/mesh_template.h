/**
 * \file fea/mesh_template.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

//! implementation of template classes
#pragma once

#include "fea/mesh.h"

#define INST_MESH_IO_TRANS(dim, Mesh)               \
    namespace fea {                                 \
    template class MeshShapeMatTrans<dim, Mesh>;    \
    template class MeshForceOutputTrans<dim, Mesh>; \
    template class DeformableBody<dim, Mesh>;       \
    }

/* ======================= MeshShapeMatTrans ======================= */
template <int dim, class Mesh>
fea::MeshShapeMatTrans<dim, Mesh>::MeshShapeMatTrans(
        MeshPtr mesh, const CoordMaskND<dim>& fixed_mask,
        const CoordMatND<dim>* init_vtx_coord, const CoordMatND<dim>* vtx_delta)
        : m_has_delta{vtx_delta != nullptr}, m_mesh{std::move(mesh)} {
    const size_t nr_vtx = m_mesh->nr_vertices();
    sanm_assert(nr_vtx == static_cast<size_t>(fixed_mask.cols()));

    size_t nr_fixed = fixed_mask.count(),
           nr_unknown_vtx = nr_vtx * dim - nr_fixed;

    if (init_vtx_coord == nullptr) {
        init_vtx_coord = &m_mesh->vertices();
    } else {
        sanm_assert(init_vtx_coord->cols() == m_mesh->vertices().cols());
    }

    if (vtx_delta) {
        sanm_assert(vtx_delta->cols() == fixed_mask.cols());
    }

    // init x0 and vtx2uidx(map from vertex to the index in the unknowns)
    Eigen::Matrix<int, dim, Eigen::Dynamic> vtx2uidx(dim, nr_vtx);
    {
        fp_t* x0_ptr = m_x0.set_shape({nr_unknown_vtx}).woptr();
        m_vertex_loc.resize(nr_unknown_vtx);
        size_t u = 0;
        for (size_t i = 0; i < nr_vtx; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                if (fixed_mask(j, i)) {
                    vtx2uidx(j, i) = -1;
                } else {
                    vtx2uidx(j, i) = u;
                    x0_ptr[u] = (*init_vtx_coord)(j, i);
                    m_vertex_loc[u].vtx = i;
                    m_vertex_loc[u].coord = j;
                    ++u;
                }
            }
        }
        sanm_assert(u == nr_unknown_vtx);
    }

    const size_t nr_faces = m_mesh->nr_faces();
    // an estimation: two inputs per element
    this->m_all_input_elem.reserve(nr_faces * dim * dim * 2);
    this->m_oidx_input_elem.resize(nr_faces * dim * dim);
    m_bias.set_shape({nr_faces, dim, dim}).fill_with_inplace(0);
    fp_t* bias_ptr = nullptr;
    if (nr_fixed) {
        bias_ptr = m_bias.rwptr();
    }

    // compute the shape matrix for each element
    for (size_t ele = 0; ele < nr_faces; ++ele) {
        size_t v0 = m_mesh->faces()(0, ele);
        // we basically expand v[dm] - v[0] in the loops below
        for (size_t dm = 1; dm <= dim; ++dm) {
            size_t vi = m_mesh->faces()(dm, ele);
            for (size_t r = 0; r < dim; ++r) {
                size_t oidx = ele * (dim * dim) + r * dim + (dm - 1);
                this->m_oidx_input_elem[oidx].first =
                        this->m_all_input_elem.size();

                // -x0[r]
                if (int uidx = vtx2uidx(r, v0); uidx < 0) {
                    bias_ptr[oidx] -= (*init_vtx_coord)(r, v0);
                } else {
                    this->m_all_input_elem.emplace_back(
                            -1._fp, static_cast<size_t>(uidx));
                }

                // +xdm[r]
                if (int uidx = vtx2uidx(r, vi); uidx < 0) {
                    bias_ptr[oidx] += (*init_vtx_coord)(r, vi);
                } else {
                    this->m_all_input_elem.emplace_back(
                            1._fp, static_cast<size_t>(uidx));
                }

                if (vtx_delta) {
                    fp_t d = (*vtx_delta)(r, vi) - (*vtx_delta)(r, v0);
                    if (d != 0) {
                        this->m_all_input_elem.emplace_back(d, nr_unknown_vtx);
                    }
                }

                this->m_oidx_input_elem[oidx].second =
                        this->m_all_input_elem.size();
            }
        }
    }
}

template <int dim, class Mesh>
sanm::TensorND fea::MeshShapeMatTrans<dim, Mesh>::copy_vtx_values(
        const CoordMatND<dim>& vtx_values) const {
    using namespace sanm;
    const size_t nr_unknown = this->nr_unknown_vtx();
    sanm_assert(m_vertex_loc.size() == nr_unknown);
    sanm_assert(static_cast<size_t>(vtx_values.cols()) ==
                m_mesh->nr_vertices());
    TensorND ret{TensorShape{nr_unknown}};
    auto wptr = ret.woptr();
    for (size_t i = 0; i < nr_unknown; ++i) {
        wptr[i] = vtx_values(m_vertex_loc[i].coord, m_vertex_loc[i].vtx);
    }
    return ret;
}

/* ======================= MeshForceOutputTrans ======================= */

template <int dim, class Mesh>
fea::MeshForceOutputTrans<dim, Mesh>::MeshForceOutputTrans(
        std::shared_ptr<InputTrans> input_trans)
        : m_input_trans{std::move(input_trans)} {
    const Mesh& mesh = m_input_trans->mesh();
    const MeshVertexReverseList& vtx_rev_list = mesh.vertex_reverse_list();
    const CoordMatND<dim>& vtx_norms = mesh.vertex_norms();
    const size_t nr_unknown = m_input_trans->nr_unknown_vtx();
    this->m_oidx_input_elem.resize(nr_unknown);
    sanm_assert(static_cast<size_t>(vtx_norms.cols()) ==
                mesh.nr_faces() * (dim + 1));

    this->m_all_input_elem.reserve(nr_unknown * 2 * dim);

    for (size_t i = 0; i < nr_unknown; ++i) {
        this->m_oidx_input_elem[i].first = this->m_all_input_elem.size();

        auto vi_loc = m_input_trans->vertex_loc()[i];
        for (auto adjacent : vtx_rev_list.query(vi_loc.vtx)) {
            Eigen::Matrix<fp_t, dim, 1> norm =
                    vtx_norms.col(adjacent.ele * (dim + 1) + adjacent.vtx_id);
            for (int j = 0; j < dim; ++j) {
                this->m_all_input_elem.emplace_back(
                        norm[j],
                        adjacent.ele * (dim * dim) + vi_loc.coord * dim + j);
            }
        }

        this->m_oidx_input_elem[i].second = this->m_all_input_elem.size();
    }
}

/* ======================= DeformableBody ======================= */

template <int dim, class Mesh>
fea::DeformableBody<dim, Mesh>::DeformableBody(const MaterialProperty& material,
                                               MeshPtr mesh)
        : m_material{material}, m_mesh{std::move(mesh)} {
    m_coord_fixed_mask = CoordMaskND<dim>::Zero(dim, m_mesh->nr_vertices());
}

template <int dim, class Mesh>
std::unique_ptr<typename fea::DeformableBody<dim, Mesh>::ElasticForceModel>
fea::DeformableBody<dim, Mesh>::make_inverse(EnergyModel energy_model) const {
    using namespace sanm;
    using namespace symbolic;
    auto ret = std::make_unique<ElasticForceModel>();
    ret->lt_inp = std::make_shared<MeshShapeMatTrans<dim, Mesh>>(
            m_mesh, m_coord_fixed_mask, nullptr, nullptr);
    ret->lt_out =
            std::make_shared<MeshForceOutputTrans<dim, Mesh>>(ret->lt_inp);
    SymbolVar Dm = placeholder(ret->cg) +
                   constant(ret->cg, ret->lt_inp->bias()),
              Ds = constant(ret->cg, m_mesh->shape_matrix()),
              F = batched_mat_inv_mul(Dm, Ds, true),
              sigma = cauchy_stress(energy_model, m_material, F, dim);
    ret->y = sigma;
    return ret;
}

template <int dim, class Mesh>
std::unique_ptr<typename fea::DeformableBody<dim, Mesh>::ElasticForceModel>
fea::DeformableBody<dim, Mesh>::make_forward(
        EnergyModel energy_model, const CoordMatND<dim>* init_vtx_coord,
        const CoordMatND<dim>* vtx_delta) const {
    using namespace sanm;
    using namespace symbolic;
    auto ret = std::make_unique<ElasticForceModel>();
    ret->lt_inp = std::make_shared<MeshShapeMatTrans<dim, Mesh>>(
            m_mesh, m_coord_fixed_mask, init_vtx_coord, vtx_delta);
    ret->lt_out =
            std::make_shared<MeshForceOutputTrans<dim, Mesh>>(ret->lt_inp);
    SymbolVar Ds = placeholder(ret->cg) +
                   constant(ret->cg, ret->lt_inp->bias()),
              DmInv = constant(ret->cg,
                               m_mesh->shape_matrix().batched_matinv()),
              F = Ds.batched_matmul(DmInv),
              P = pk1(energy_model, m_material, F, dim),
              potential_density = elastic_potential_density(energy_model,
                                                            m_material, F, dim);
    ret->y = P;
    if (potential_density.node()) {
        auto& vol_vec = m_mesh->face_areas();
        TensorND volumes{{vol_vec.size(), 1}};
        memcpy(volumes.woptr(), vol_vec.data(), sizeof(fp_t) * vol_vec.size());
        ret->potential = potential_density * constant(ret->cg, volumes);
    }
    return ret;
}

template <int dim, class Mesh>
fea::fp_t fea::DeformableBody<dim, Mesh>::compute_force_rms(
        const ElasticForceModel& model, const sanm::TensorND& xt,
        const sanm::TensorND& f_load, const Mesh& final_mesh,
        bool sanity_check) {
    sanm::TensorND sym_inpval = model.lt_inp->apply(xt),
                   shape_mat0 = sym_inpval + model.lt_inp->bias();
    shape_mat0.assert_allclose("shape matrix check", final_mesh.shape_matrix());
    sanm::TensorND stress_tensor = sanm::symbolic::eval_unary_func(
                           model.y.node(), sym_inpval),
                   internal_force = model.lt_out->apply(stress_tensor);
    if (sanity_check) {
        internal_force.assert_allclose("force equilibrium check", -f_load,
                                       1e-5);
    }
    return (internal_force + f_load).norm_rms();
}
