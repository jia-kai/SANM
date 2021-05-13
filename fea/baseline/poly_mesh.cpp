#include "poly_mesh.h"

#include <cfloat>

using namespace materials;

template <int vertex_dim, typename T>
void PolyMesh<vertex_dim, T>::GetBoundingBox(
        std::pair<VectorDimT, VectorDimT>& bounding_box) const {
    // loop over all vertices
    // find min and max in each direction
    for (int i = 0; i < vertex_dim; ++i) {
        T min_dim = DBL_MAX;
        T max_dim = -DBL_MAX;

        for (int j = 0; j < NumOfVertex(); j++) {
            T val = vertex_(i, j);
            if (val < min_dim) {
                min_dim = val;
            }
            if (val > max_dim) {
                max_dim = val;
            }
            bounding_box.first(i) = min_dim;
            bounding_box.second(i) = max_dim;
        }
    }
}

template <int vertex_dim, typename T>
void PolyMesh<vertex_dim, T>::GetScale(VectorDimT& scale) const {
    std::pair<VectorDimT, VectorDimT> bounding_box;
    GetBoundingBox(bounding_box);

    scale = bounding_box.second - bounding_box.first;
}

template <int vertex_dim, typename T>
typename PolyMesh<vertex_dim, T>::MatrixDimXT
PolyMesh<vertex_dim, T>::vertex_in_element(const int index) const {
    assert(index >= 0 && index < static_cast<int>(element_.cols()));
    const int e_dim = static_cast<int>(element_.rows());
    MatrixDimXT v = MatrixXT::Zero(vertex_dim, e_dim);
    for (int i = 0; i < e_dim; ++i) {
        v.col(i) = vertex_.col(element_(i, index));
    }
    return v;
}

template class materials::PolyMesh<3, double>;
