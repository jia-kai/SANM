#pragma once

#include <Eigen/Dense>

#include "utils.h"

namespace materials {

/*!
 * a base mesh class, which consists of:
 *  * vertex_: a m x n matrix where each column is a m-dimensional point.
 *  * element_: a p x q matrix where each column represents an element in the
 *             mesh.
 *  * edge_in_elemnet_: a 2 x r matrix which defines the edges in an element.
 *
 *  Example:
 *  * A 2-D square represented by a triangle mesh:
 *      vertex_ = [0 0; 0 1; 1 0; 1 1]';
 *      element_ = [0 1 2; 1 2 3]';
 *      edge_in_element_ = [0 1; 0 2; 1 2]';
 *
 *  * A 2-D square represented by a quad mesh:
 *      vertex_ = [0 0; 0 1; 1 0; 1 1]';
 *      element_ = [0 1 3 2]';
 *      edge_in_element_ = [0 1; 1 2; 2 3; 3 0]';
 */
template <int vertex_dim, typename T>
class PolyMesh {
protected:
    // Name convention: replace the number with 'Dim' in Matrix2Xd, Vector2d,
    // Vector2i.
    typedef Eigen::Matrix<T, vertex_dim, Eigen::Dynamic> MatrixDimXT;
    typedef Eigen::Matrix<T, vertex_dim, 1> VectorDimT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;

public:
    PolyMesh(const MatrixDimXT& vertex, const Eigen::MatrixXi& element,
             const Eigen::Matrix2Xi& edge_in_element)
            : vertex_(vertex),
              element_(element),
              edge_in_element_(edge_in_element) {}

    virtual ~PolyMesh() = default;

    const MatrixDimXT& vertex() const { return vertex_; }
    VectorDimT vertex(const int index) const {
        cf_assert(index >= 0 && index < static_cast<int>(vertex_.cols()));
        return vertex_.col(index);
    }
    const Eigen::MatrixXi& element() const { return element_; }
    Eigen::VectorXi element(const int index) const {
        cf_assert(index >= 0 && index < static_cast<int>(element_.cols()));
        return element_.col(index);
    }
    const Eigen::Matrix2Xi& edge_in_element() const { return edge_in_element_; }
    MatrixDimXT vertex_in_element(const int index) const;
    int NumOfVertex() const { return static_cast<int>(vertex_.cols()); }
    int NumOfElement() const { return static_cast<int>(element_.cols()); }

    void GetBoundingBox(std::pair<VectorDimT, VectorDimT>& bounding_box) const;
    void GetScale(VectorDimT& scale) const;

protected:
    MatrixDimXT vertex_;
    Eigen::MatrixXi element_;
    Eigen::Matrix2Xi edge_in_element_;
};

}  // namespace materials
