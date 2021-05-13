#include "tet_elastic_body.h"
#include "material.h"
#include "utils.h"

using namespace materials;

template <typename T>
TetElasticBody<T>::TetElasticBody(const Material<3, T>& material,
                                  const Matrix3X<T>& initial_vertex_position,
                                  const T density,
                                  const TetMesh<T>& undeformed_tet_mesh)
        : ElasticBody<3, T>(material, initial_vertex_position,
                            undeformed_tet_mesh) {
    // initialization
    // pre-compute constant values Dm_inv, dFdx and tet_volume for each tet
    // element
    m_Dm_inv.clear();
    m_dFdx.clear();
    m_tet_volume.clear();
    for (int i = 0; i < this->m_undeformed_mesh.NumOfElement(); i++) {
        const Eigen::Matrix<int, 4, 1>& elements =
                this->m_undeformed_mesh.element(i);
        Eigen::Matrix<T, 3, 3> Dm;
        // Dm = [x0 - x3, x1 - x3, x2 - x3]
        for (int j = 0; j < 3; j++) {
            Dm.col(j) = this->m_undeformed_mesh.vertex(elements[j]) -
                        this->m_undeformed_mesh.vertex(elements[3]);
        }
        m_Dm_inv.push_back(Dm.inverse());
        Eigen::Matrix<T, 9, 12> dFdx_i;
        dFdx_i.setZero();
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                // compute d Ds / d x[j, k]
                // xs are flattened from a col-major matrix [x0, x1, x2, x3]
                Eigen::Matrix<T, 3, 3> dDsdx;
                dDsdx.setZero();
                if (j < 3) {
                    dDsdx(k, j) = 1;
                } else {
                    for (int l = 0; l < 3; l++)
                        dDsdx(k, l) = -1;
                }
                Eigen::Matrix<T, 3, 3> dFdx_mat = dDsdx * m_Dm_inv[i];
                for (int s = 0; s < 3; s++)
                    for (int t = 0; t < 3; t++)
                        dFdx_i(s * 3 + t, j * 3 + k) = dFdx_mat(t, s);
            }
        }
        m_dFdx.push_back(dFdx_i);
        m_tet_volume.push_back(std::fabs(Dm.determinant()) / 6.0);
    }
}

template <typename T>
TetElasticBody<T>::~TetElasticBody() = default;

template <typename T>
Matrix3X<T> TetElasticBody<T>::compute_force(
        const Matrix3X<T>& vertices) const {
    Matrix3X<T> force = MatrixX<T>::Zero(3, vertices.cols());
    const int tet_num = this->m_undeformed_mesh.NumOfElement();
    for (int i = 0; i < tet_num; i++) {
        const Eigen::Matrix<int, 4, 1> elements =
                this->m_undeformed_mesh.element(i);
        const Material<3, T>& material =
                this->m_materials[this->m_material_id[i]].get();
        Eigen::Matrix<T, 3, 3> Ds;  // deformed shape matrix
        for (int j = 0; j < 3; j++)
            Ds.col(j) = vertices.col(elements[j]) - vertices.col(elements[3]);
        Eigen::Matrix<T, 3, 3> F = Ds * m_Dm_inv[i];
        Eigen::Matrix<T, 3, 3> P = material.StressTensor(F);
        Eigen::Matrix<T, 12, 1> df;
        df.setZero();
        for (int j = 0; j < 12; j++) {
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    df(j, 0) += P(k, l) * m_dFdx[i](l * 3 + k, j);
        }
        for (int j = 0; j < 4; j++)
            force.col(elements[j]) -= df.segment(3 * j, 3) * m_tet_volume[i];
    }
    return force;
}

template <typename T>
T TetElasticBody<T>::compute_energy(const Matrix3X<T>& vertices) const {
    const int tet_num = this->m_undeformed_mesh.NumOfElement();
    T energy = 0;
    for (int i = 0; i < tet_num; i++) {
        const Eigen::Matrix<int, 4, 1> elements =
                this->m_undeformed_mesh.element(i);
        const Material<3, T>& material =
                this->m_materials[this->m_material_id[i]].get();
        Eigen::Matrix<T, 3, 3> Ds;  // deformed shape matrix
        for (int j = 0; j < 3; j++)
            Ds.col(j) = vertices.col(elements[j]) - vertices.col(elements[3]);
        Eigen::Matrix<T, 3, 3> F = Ds * m_Dm_inv[i];
        energy += material.EnergyDensity(F) * m_tet_volume[i];
    }
    return energy;
}

template <typename T>
typename TetElasticBody<T>::SparseMat
TetElasticBody<T>::compute_stiffness_matrix(const Matrix3X<T>& vertices) const {
    cf_assert(vertices.cols() == this->m_undeformed_mesh.NumOfVertex());
    // triplet_list is a vector of triplet <row, col, value>
    std::vector<Eigen::Triplet<T>> triplet_list;
    const int vertex_num = static_cast<int>(vertices.cols());
    const int tet_num = this->m_undeformed_mesh.NumOfElement();
    triplet_list.reserve(tet_num * 4);
    // compute Ki for each tet element and fill the triplet_list vector
    for (int i = 0; i < tet_num; i++) {
        const Eigen::VectorXi& elements = this->m_undeformed_mesh.element(
                i);  // get the element of the tet
        const Material<3, T>& material =
                this->m_materials[this->m_material_id[i]]
                        .get();  // get the material model

        /* Implement your code here */
        Eigen::Matrix<T, 3, 3> Ds;
        for (int j = 0; j < 3; j++) {
            Ds.col(j) = vertices.col(elements[j]) - vertices.col(elements[3]);
        }
        const Eigen::Matrix<T, 3, 3> F = Ds * m_Dm_inv[i];
        const Eigen::Matrix<T, 9, 9> dPdF = material.StressDifferential(F);
        const Eigen::Matrix<T, 9, 12>&dFdx = m_dFdx[i], dPdx = dPdF * dFdx;

        for (int fi = 0; fi < 4; ++fi) {
            for (int fj = 0; fj < 3; ++fj) {
                for (int xi = 0; xi < 4; ++xi) {
                    for (int xj = 0; xj < 3; ++xj) {
                        T d = -dPdx.col(xi * 3 + xj)
                                       .dot(dFdx.col(fi * 3 + fj)) *
                              m_tet_volume[i];
                        int vi = elements[fi] * 3 + fj,
                            vj = elements[xi] * 3 + xj;
                        triplet_list.emplace_back(vi, vj, -d);
                    }
                }
            }
        }
    }
    SparseMat K(vertex_num * 3, vertex_num * 3);
    // contruct sparse matrix K from triplet list, K(i, j) is sum of all value
    // of triplet with row = i and col = j
    K.setFromTriplets(triplet_list.begin(), triplet_list.end());
    // ensure a self-adjoint matrix
    K = (K + SparseMat(K.transpose())) / 2.0;
    return K;
}

template class materials::TetElasticBody<double>;
