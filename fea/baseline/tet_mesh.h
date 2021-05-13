#pragma once

#include "poly_mesh.h"
#include "typedefs.h"

#include <vector>

namespace materials {
template <typename T>
class TetMesh : public PolyMesh<3, T> {
public:
    TetMesh(const Matrix3X<T>& vertex, const Matrix4Xi& element)
            : PolyMesh<3, T>(vertex, element,
                             (Eigen::Matrix<int, 2, 6>() << 0, 0, 0, 1, 1, 2, 1,
                              2, 3, 2, 3, 3)
                                     .finished()) {}

    TetMesh(const TetMesh& tet_mesh) : PolyMesh<3, T>(tet_mesh) {}

    TetMesh& operator=(const TetMesh& tet_mesh) {
        PolyMesh<3, T>::operator=(tet_mesh);
        return *this;
    }
};
}  // namespace materials
