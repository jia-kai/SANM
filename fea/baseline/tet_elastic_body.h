#pragma once

#include "elastic_body.h"
#include "tet_mesh.h"
#include "typedefs.h"

namespace materials {
template <typename T>
class TetElasticBody : public ElasticBody<3, T> {
    std::vector<Eigen::Matrix<T, 3, 3>> m_Dm_inv;
    std::vector<Eigen::Matrix<T, 9, 12>> m_dFdx;
    std::vector<T> m_tet_volume;

public:
    using typename ElasticBody<3, T>::SparseMat;

    TetElasticBody& operator=(const TetElasticBody&) = delete;

    TetElasticBody(const Material<3, T>& material,
                   const Matrix3X<T>& initial_vertex_position, const T density,
                   const TetMesh<T>& undeformed_tet_mesh);

    ~TetElasticBody();

    Matrix3X<T> compute_force(const Matrix3X<T>& vertices) const;

    T compute_energy(const Matrix3X<T>& vertices) const;

    SparseMat compute_stiffness_matrix(
            const Matrix3X<T>& vertices) const override;
};

}  // namespace materials
