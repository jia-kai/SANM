/**
 * \file fea/tetrahedral_mesh.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "fea/tetrahedral_mesh.h"
#include "fea/mesh_template.h"

#include <Eigen/Dense>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>

using namespace fea;

INST_MESH_IO_TRANS(3, TetrahedralMesh);

namespace {
void copy_mat3_cols(fp_t* dst, const Vec3& c0, const Vec3& c1, const Vec3& c2) {
    const Vec3* cs[] = {&c0, &c1, &c2};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            dst[i * 3 + j] = (*cs[j])[i];
        }
    }
}
}  // anonymous namespace

const CoordMat3D& TetrahedralMesh::vertex_norms() const {
    if (m_vertex_norms.valid()) {
        return m_vertex_norms.val();
    }
    size_t nr_tet = this->nr_tet();
    m_vertex_norms.init();
    CoordMat3D& vertex_norms = m_vertex_norms.val();
    vertex_norms.resize(3, nr_tet * 4);
    m_tet_volumes.resize(nr_tet);
    auto ds_ptr = m_shape_matrix.set_shape({nr_tet, 3, 3}).woptr();

    for (size_t i = 0; i < nr_tet; ++i) {
        Vec3 x0 = m_vertices.col(m_tet(0, i)), x1 = m_vertices.col(m_tet(1, i)),
             x2 = m_vertices.col(m_tet(2, i)), x3 = m_vertices.col(m_tet(3, i)),
             v1 = x1 - x0, v2 = x2 - x0, v3 = x3 - x0;
        copy_mat3_cols(ds_ptr + i * 9, v1, v2, v3);

        fp_t det = v1.dot(v2.cross(v3));

        m_tet_volumes[i] = std::fabs(det) / 6;

        // - volume * D^{-T}, equal to -cofactor*sign(det)
        // normal for a vertex is also the area-weighted outward normal of the
        // opposite face

        Vec3 t1 = v2.cross(v3), t2 = v3.cross(v1), t3 = v1.cross(v2);
        if (det > 0) {
            t1 = -t1;
            t2 = -t2;
            t3 = -t3;
        }
        vertex_norms.col(4 * i + 0) = -(t1 + t2 + t3);
        vertex_norms.col(4 * i + 1) = t1;
        vertex_norms.col(4 * i + 2) = t2;
        vertex_norms.col(4 * i + 3) = t3;
    }
    vertex_norms *= 1.0 / 6;
    return vertex_norms;
}

const std::vector<fp_t>& TetrahedralMesh::tet_volumes() const {
    if (m_tet_volumes.empty()) {
        vertex_norms();
    }
    return m_tet_volumes;
}

const sanm::TensorND& TetrahedralMesh::shape_matrix() const {
    if (m_shape_matrix.empty()) {
        vertex_norms();
    }
    return m_shape_matrix;
}

const MeshVertexReverseList& TetrahedralMesh::vertex_reverse_list() const {
    if (!m_vertex_reverse_list.valid()) {
        m_vertex_reverse_list.init(
                MeshVertexReverseList::from_mesh<4>(nr_vertices(), m_tet));
    }
    return m_vertex_reverse_list.val();
}

TetrahedralMeshPtr TetrahedralMesh::make_cuboid(size_t nr_vtx_x,
                                                size_t nr_vtx_y,
                                                size_t nr_vtx_z, fp_t size) {
    sanm_assert(nr_vtx_x >= 2 && nr_vtx_y >= 2 && nr_vtx_z >= 2 && size > 0);
    // code adopted from the CompFabAssignment
    const int vertex_num = nr_vtx_x * nr_vtx_y * nr_vtx_z;
    const int element_num =
            5 * (nr_vtx_x - 1) * (nr_vtx_y - 1) * (nr_vtx_z - 1);
    auto ret = std::make_shared<TetrahedralMesh>();
    CoordMat3D& vertex = ret->m_vertices;
    TetIndexMat& element = ret->m_tet;
    vertex.resize(3, vertex_num);
    element.resize(4, element_num);
    int id = 0;
    for (size_t i = 0; i < nr_vtx_x; i++) {
        for (size_t j = 0; j < nr_vtx_y; j++) {
            for (size_t k = 0; k < nr_vtx_z; k++) {
                vertex(0, id) = i * size;
                vertex(1, id) = j * size;
                vertex(2, id) = k * size;
                if (i == 0 || i == nr_vtx_x - 1 || j == 0 ||
                    j == nr_vtx_y - 1 || k == 0 || k == nr_vtx_z - 1) {
                    ret->m_surface_vtx.insert(id);
                }
                id++;
            }
        }
    }
    auto get_cuboid_id = [nr_vtx_y, nr_vtx_z](size_t x, size_t y,
                                              size_t z) -> int {
        return (x * nr_vtx_y + y) * nr_vtx_z + z;
    };
    id = 0;
    for (size_t i = 0; i < nr_vtx_x - 1; i++) {
        for (size_t j = 0; j < nr_vtx_y - 1; j++) {
            for (size_t k = 0; k < nr_vtx_z - 1; k++) {
                int hex_id[] = {get_cuboid_id(i, j, k),
                                get_cuboid_id(i + 1, j, k),
                                get_cuboid_id(i + 1, j + 1, k),
                                get_cuboid_id(i, j + 1, k),
                                get_cuboid_id(i, j, k + 1),
                                get_cuboid_id(i + 1, j, k + 1),
                                get_cuboid_id(i + 1, j + 1, k + 1),
                                get_cuboid_id(i, j + 1, k + 1)};
                auto add_face = [&](int a, int b, int c) {
                    auto& t = ret->m_surfaces.emplace_back();
                    t[0] = hex_id[a];
                    t[1] = hex_id[b];
                    t[2] = hex_id[c];
                };

                if (i == 0) {
                    add_face(3, 0, 7);
                    add_face(7, 0, 4);
                }
                if (i == nr_vtx_x - 2) {
                    add_face(1, 2, 6);
                    add_face(6, 5, 1);
                }
                if (j == 0) {
                    add_face(0, 1, 5);
                    add_face(0, 5, 4);
                }
                if (j == nr_vtx_y - 2) {
                    add_face(7, 6, 3);
                    add_face(6, 2, 3);
                }
                if (k == 0) {
                    add_face(1, 3, 2);
                    add_face(0, 3, 1);
                }
                if (k == nr_vtx_z - 2) {
                    add_face(4, 5, 7);
                    add_face(7, 5, 6);
                }

                // 0, 2, 1, 5
                element(0, id) = hex_id[0];
                element(1, id) = hex_id[2];
                element(2, id) = hex_id[1];
                element(3, id) = hex_id[5];
                id++;
                // 0, 4, 7, 5
                element(0, id) = hex_id[0];
                element(1, id) = hex_id[4];
                element(2, id) = hex_id[7];
                element(3, id) = hex_id[5];
                id++;
                // 0, 2, 5, 7
                element(0, id) = hex_id[0];
                element(1, id) = hex_id[2];
                element(2, id) = hex_id[5];
                element(3, id) = hex_id[7];
                id++;
                // 2, 6, 5, 7
                element(0, id) = hex_id[2];
                element(1, id) = hex_id[6];
                element(2, id) = hex_id[5];
                element(3, id) = hex_id[7];
                id++;
                // 0, 7, 3, 2
                element(0, id) = hex_id[0];
                element(1, id) = hex_id[7];
                element(2, id) = hex_id[3];
                element(3, id) = hex_id[2];
                id++;
            }
        }
    }
    sanm_assert(id == element_num);
    return ret;
}

TetrahedralMeshPtr TetrahedralMesh::from_tetgen_files(
        const std::string& filebase) {
    std::ifstream fin_ele{filebase + ".ele"}, fin_node{filebase + ".node"},
            fin_face{filebase + ".face"};
    sanm_assert(fin_ele.good() && fin_node.good() && fin_face.good(),
                "failed to open input files: %s.{ele,node,face}",
                filebase.c_str());
    auto ret = std::make_shared<TetrahedralMesh>();

    // see https://wias-berlin.de/software/tetgen/fformats.node.html
    size_t nr_vtx, dim, nr_attr, bound_mark;
    fin_node >> nr_vtx >> dim >> nr_attr >> bound_mark;
    sanm_assert(dim == 3 && !nr_attr && !bound_mark);

    CoordMat3D& vtx = ret->m_vertices;
    vtx.resize(3, nr_vtx);
    for (size_t i = 0; i < nr_vtx; ++i) {
        size_t idx;
        fin_node >> idx >> vtx(0, i) >> vtx(1, i) >> vtx(2, i);
        sanm_assert(idx == i, "failed to read vertex %zu: got idx %zu", i, idx);
    }
    sanm_assert(fin_node.good());

    // see https://wias-berlin.de/software/tetgen/fformats.ele.html
    size_t nr_tet, node_per_tet;
    fin_ele >> nr_tet >> node_per_tet >> nr_attr;
    sanm_assert(node_per_tet == 4 && !nr_attr);

    TetIndexMat& tet = ret->m_tet;
    tet.resize(4, nr_tet);
    for (size_t i = 0; i < nr_tet; ++i) {
        size_t idx;
        fin_ele >> idx >> tet(0, i) >> tet(1, i) >> tet(2, i) >> tet(3, i);
        sanm_assert(idx == i, "failed to read tetrahedron %zu: got idx %zu", i,
                    idx);
    }

    size_t nr_face, boundary_marker;
    fin_face >> nr_face >> boundary_marker;
    for (size_t i = 0; i < nr_face; ++i) {
        size_t idx;
        int a, b, c;
        fin_face >> idx >> a >> b >> c;
        sanm_assert(idx == i);
        ret->m_surface_vtx.insert(a);
        ret->m_surface_vtx.insert(b);
        ret->m_surface_vtx.insert(c);
        if (boundary_marker) {
            fin_face >> b;
        }
        // do not read into m_surfaces since tetgen may invert the surface
    }

    return ret;
}

void TetrahedralMesh::write_to_file(FILE* fout,
                                    const VertexSet* filter_set) const {
    if (!filter_set) {
        if (!m_surfaces.empty()) {
            write_to_file(fout, m_vertices, m_surfaces);
            return;
        }

        if (!m_surface_vtx.empty()) {
            filter_set = &m_surface_vtx;
        }
    }
    write_to_file(fout, m_vertices, m_tet, filter_set);
}

void TetrahedralMesh::write_to_surface_vtx_file(FILE* fout) const {
    sanm_assert(!m_surface_vtx.empty());
    int vmin = std::numeric_limits<int>::max(), vmax = 0;
    for (int i : m_surface_vtx) {
        vmin = std::min(vmin, i);
        vmax = std::max(vmax, i);
    }
    sanm_assert(vmin == 0, "min surface vtx num is not zero: %d", vmin);
    sanm_assert(vmax == static_cast<int>(m_surface_vtx.size()) - 1,
                "max surface vtx num is %d, size is %zu", vmax,
                m_surface_vtx.size());

    for (int i = vmin; i <= vmax; ++i) {
        const Vec3& v = m_vertices.col(i);
        fprintf(fout, "%g %g %g\n", v[0], v[1], v[2]);
    }
}

void TetrahedralMesh::write_to_file(FILE* fout, const CoordMat3D& V,
                                    const TetIndexMat& F,
                                    const VertexSet* filter_set) {
    sanm_assert(fout);
    std::unordered_map<int, int> vtx_id_map;
    if (filter_set) {
        sanm_assert(!filter_set->empty());
    }
    auto write_facet = [fout, filter_set, &vtx_id_map](int v0, int v1, int v2) {
        if (filter_set) {
            if (!filter_set->count(v0) || !filter_set->count(v1) ||
                !filter_set->count(v2)) {
                return;
            }
            v0 = vtx_id_map.at(v0);
            v1 = vtx_id_map.at(v1);
            v2 = vtx_id_map.at(v2);
        }
        fprintf(fout, "f %d %d %d\n", v0 + 1, v1 + 1, v2 + 1);
    };
    for (int i = 0; i < V.cols(); ++i) {
        if (!filter_set || filter_set->count(i)) {
            if (filter_set) {
                int id = vtx_id_map.size();
                vtx_id_map[i] = id;
            }
            Vec3 vi = V.col(i);
            fprintf(fout, "v %g %g %g\n", vi.x(), vi.y(), vi.z());
        }
    }

    for (Eigen::Index i = 0; i < F.cols(); ++i) {
        int i0 = F(0, i), i1 = F(1, i), i2 = F(2, i), i3 = F(3, i);
        Vec3 v0 = V.col(i0), v1 = V.col(i1), v2 = V.col(i2), v3 = V.col(i3);

        if ((v1 - v0).dot((v2 - v0).cross(v3 - v0)) > 0) {
            std::swap(i1, i2);
        }

        write_facet(i0, i1, i2);
        write_facet(i1, i3, i2);
        write_facet(i1, i0, i3);
        write_facet(i0, i2, i3);
    }
}

void TetrahedralMesh::write_to_file(FILE* fout, const CoordMat3D& V,
                                    const FaceList& F) {
    sanm_assert(fout && !F.empty());
    std::unordered_map<int, int> vtx_id_map;
    std::vector<int> vtx_ids;
    vtx_ids.reserve(F.size() * 2);
    vtx_id_map.reserve(F.size() * 2);

    for (auto&& f : F) {
        for (int v : f) {
            int id = vtx_id_map.size();
            if (vtx_id_map.insert({v, id}).second) {
                vtx_ids.push_back(v);
            }
        }
    }

    for (int i : vtx_ids) {
        Vec3 vi = V.col(i);
        fprintf(fout, "v %g %g %g\n", vi.x(), vi.y(), vi.z());
    }

    for (auto&& f : F) {
        int v0 = vtx_id_map.at(f[0]), v1 = vtx_id_map.at(f[1]),
            v2 = vtx_id_map.at(f[2]);
        fprintf(fout, "f %d %d %d\n", v0 + 1, v1 + 1, v2 + 1);
    }
}

void TetrahedralMesh::replace_with_mask(const CoordMask3D& mask,
                                        const sanm::TensorND& value) {
    fea::replace_with_mask(m_vertices, mask, value.ptr(),
                           value.shape().total_nr_elems());
    clear_cache();
}

void TetrahedralMesh::apply_vtx_delta(const CoordMat3D& delta) {
    sanm_assert(delta.cols() == m_vertices.cols());
    m_vertices += delta;
    clear_cache();
}

void TetrahedralMesh::replace_vtx(const CoordMat3D& vtx) {
    sanm_assert(vtx.cols() == m_vertices.cols());
    m_vertices = vtx;
    clear_cache();
}

void TetrahedralMesh::clear_cache() {
    m_vertex_norms.reset();
    m_tet_volumes.clear();
    m_shape_matrix.clear();
}

void TetrahedralMesh::resize_inplace(fp_t scale) {
    m_vertices *= scale;
    clear_cache();
}
