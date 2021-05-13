#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "poly_mesh.h"
#include "typedefs.h"

namespace materials {
template <int dim, typename T>
class Material;

template <int dim, typename T>
class ElasticBody {
public:
    // Type name convention: replace number in "Dim" in Vector2d or Vector3d.
    typedef Eigen::Matrix<T, dim, 1> VectorDimT;
    typedef Eigen::Matrix<T, dim, Eigen::Dynamic> MatrixDimXT;
    typedef PolyMesh<dim, T> PolyMeshDim;

    virtual ~ElasticBody() {}

    using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;

    const MatrixDimXT& vertex_position() const { return m_vertex_position; }

    // The return matrix must be symmetric and compressed. Assume A is the
    // stiffness matrix, then we can return (A.transpose() +
    // A).makeCompressed(), or use prune() to convert it into compressed format.
    virtual SparseMat compute_stiffness_matrix(
            const MatrixDimXT& vertices) const = 0;

    const PolyMeshDim& GetUndeformedMesh() const { return m_undeformed_mesh; }

protected:
    // We allow users to specify different materials for each element. If a
    // single material is provided then m_materials.size() = 1 and
    // m_material_id[i] = 0 for all i. Otherwise materials[m_material_id[i]] is
    // the material in element i.
    const std::vector<std::reference_wrapper<const Material<dim, T>>>
            m_materials;
    const std::vector<int> m_material_id;

    // Undeformed mesh.
    const PolyMeshDim& m_undeformed_mesh;

    MatrixDimXT m_vertex_position;

    ElasticBody(const Material<dim, T>& material,
                const MatrixDimXT& vertex_position,
                const PolyMeshDim& undeformed_mesh)
            : m_materials(1, material),
              m_material_id(undeformed_mesh.NumOfElement(), 0),
              m_undeformed_mesh(undeformed_mesh),
              m_vertex_position(vertex_position) {}

    ElasticBody(
            const std::vector<std::reference_wrapper<const Material<dim, T>>>&
                    materials,
            const std::vector<int>& material_id,
            const MatrixDimXT& vertex_position,
            const PolyMeshDim& undeformed_mesh)
            : m_materials(materials),
              m_material_id(material_id),
              m_undeformed_mesh(undeformed_mesh),
              m_vertex_position(vertex_position) {}

    ElasticBody(const ElasticBody<dim, T>& other)
            : m_materials(other.m_materials),
              m_material_id(other.m_material_id),
              m_undeformed_mesh(other.m_undeformed_mesh),
              m_vertex_position(other.m_vertex_position) {}

private:
    // Disable the copy assignment because we have constant data members.
    ElasticBody<dim, T>& operator=(const ElasticBody<dim, T>&);
};

}  // namespace materials

