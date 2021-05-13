/**
 * \file fea/tetrahedral_mesh.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "fea/material.h"
#include "fea/mesh.h"
#include "libsanm/anm.h"

#include <Eigen/Core>
#include <unordered_set>

namespace fea {
using CoordMat3D = CoordMatND<3>;
using CoordMask3D = CoordMaskND<3>;
using TetIndexMat = IndexMat<4>;

class TetrahedralMesh;
using TetrahedralMeshPtr = std::shared_ptr<TetrahedralMesh>;

//! representation of tetrahedral mesh in 3D space
class TetrahedralMesh {
public:
    using VertexSet = std::unordered_set<int>;
    using FaceList = std::vector<std::array<int, 3>>;

    TetrahedralMesh() = default;
    TetrahedralMesh(const CoordMat3D& vertices, const TetIndexMat& tet)
            : m_vertices{vertices}, m_tet{tet} {}

    TetrahedralMesh(const TetrahedralMesh& rhs)
            : m_vertices{rhs.m_vertices},
              m_tet{rhs.m_tet},
              m_surface_vtx{rhs.m_surface_vtx},
              m_surfaces{rhs.m_surfaces} {}

    //! number of vertices
    size_t nr_vertices() const { return m_vertices.cols(); }

    //! number of tetrahedrons
    size_t nr_tet() const { return m_tet.cols(); }
    size_t nr_faces() const { return nr_tet(); }

    //! volume of each tetrahedron
    const std::vector<fp_t>& tet_volumes() const;
    const std::vector<fp_t>& face_areas() const { return tet_volumes(); }

    //! coordinates of the vertices
    const CoordMat3D& vertices() const { return m_vertices; }

    //! vertex ID of the tetrahedrons
    const TetIndexMat& tetrahedrons() const { return m_tet; }
    const TetIndexMat& faces() const { return m_tet; }

    //! get the norms of each vertex in each tetrahedron, in nr_tet*4 rows
    const CoordMat3D& vertex_norms() const;

    //! the shape matrix matrices: xi-x0
    const sanm::TensorND& shape_matrix() const;

    //! get the reverse list that maps vertices to tetrahedrons
    const MeshVertexReverseList& vertex_reverse_list() const;

    /*!
     * \brief write the mesh to an ascii obj file
     * \param filter_set if provided, only write triangles contained in this set
     */
    void write_to_file(FILE* fout, const VertexSet* filter_set = nullptr) const;

    //! write new surface vertex coordinates to a file; the surface vertex
    //! numbers must be consecutive integers starting from 1.
    void write_to_surface_vtx_file(FILE* fout) const;

    static void write_to_file(FILE* fout, const CoordMat3D& V,
                              const TetIndexMat& F,
                              const VertexSet* filter_set = nullptr);

    static void write_to_file(FILE* fout, const CoordMat3D& V,
                              const FaceList& F);
    /*!
     * \brief replace vertex coordinates with given mask
     * \param clear_shape_mat whether to clear computed shape matrix and norms
     */
    void replace_with_mask(const CoordMask3D& mask,
                           const sanm::TensorND& value);

    //! add given delta to vertex coordinates
    void apply_vtx_delta(const CoordMat3D& delta);

    //! copy vertex coordinates from another mat
    void replace_vtx(const CoordMat3D& vtx);

    //! scale the vertice coordinates inplace; cached norms will be cleared
    void resize_inplace(fp_t scale);

    //! make a cuboid mesh
    static TetrahedralMeshPtr make_cuboid(size_t nr_vtx_x, size_t nr_vtx_y,
                                          size_t nr_vtx_z, fp_t size);

    /*!
     * \brief parse TetGen .node and .ele files to create a mesh
     * \param filebase file basename, excluding the .node or .ele part
     */
    static TetrahedralMeshPtr from_tetgen_files(const std::string& filebase);

    const FaceList& surface_list() const { return m_surfaces; }

    const VertexSet& surface_vtx() const { return m_surface_vtx; }

    const VertexSet* surface_vtx_ptr() const {
        return m_surface_vtx.empty() ? nullptr : &m_surface_vtx;
    }

private:
    CoordMat3D m_vertices;
    TetIndexMat m_tet;
    VertexSet m_surface_vtx;  //!< vertices on the surface
    FaceList m_surfaces;      //!< boundary face list

    mutable sanm::Maybe<CoordMat3D> m_vertex_norms;
    mutable sanm::TensorND m_shape_matrix;
    mutable std::vector<fp_t> m_tet_volumes;
    mutable sanm::Maybe<MeshVertexReverseList> m_vertex_reverse_list;

    TetrahedralMesh& operator=(const TetrahedralMesh& rhs) = delete;

    void clear_cache();
};

using TetShapeMatTrans = MeshShapeMatTrans<3, TetrahedralMesh>;
using TetForceOutputTrans = MeshForceOutputTrans<3, TetrahedralMesh>;
using TetrahedralDeformableBody = DeformableBody<3, TetrahedralMesh>;

using ElasticForceModel = TetrahedralDeformableBody::ElasticForceModel;

}  // namespace fea
