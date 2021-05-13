/**
 * \file fea/mesh.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "fea/mesh.h"
#include "libsanm/utils.h"

#include <algorithm>

using namespace fea;

/* ======================= global ======================= */
void fea::replace_with_mask(fp_t* dst, const bool* fixed_mask, const fp_t* src,
                            size_t dst_size, size_t src_size) {
    size_t spos = 0;
    for (size_t i = 0; i < dst_size; ++i) {
        if (!fixed_mask[i]) {
            dst[i] = src[spos++];
        }
    }
    sanm_assert(!src_size || src_size == spos,
                "src size mismatch: expect %zu, unfixed %zu", src_size, spos);
}

/* ======================= MeshVertexReverseList ======================= */
template <int msize>
MeshVertexReverseList MeshVertexReverseList::from_mesh(
        size_t nr_vtx, const IndexMat<msize>& mesh) {
    MeshVertexReverseList ret;
    std::vector<std::vector<Item>> biglist(nr_vtx);

    for (size_t i = 0; i < static_cast<size_t>(mesh.cols()); ++i) {
        for (size_t j = 0; j < msize; ++j) {
            biglist.at(mesh(j, i)).emplace_back(i, j);
        }
    }
    ret.m_results.resize(mesh.cols() * msize);
    ret.m_vtx2loc.resize(nr_vtx);
    size_t begin = 0;
    for (size_t i = 0; i < nr_vtx; ++i) {
        sanm_assert(!biglist[i].empty(), "dangling vertex %zu", i);
        std::copy_n(biglist[i].begin(), biglist[i].size(),
                    ret.m_results.data() + begin);
        ret.m_vtx2loc[i].first = begin;
        begin += biglist[i].size();
        ret.m_vtx2loc[i].second = begin;
    }
    sanm_assert(begin == ret.m_results.size());
    return ret;
}

MeshVertexReverseList::ItemSpan MeshVertexReverseList::query(size_t vtx) const {
    auto p = m_vtx2loc.at(vtx);
    auto b = m_results.data();
    return {b + p.first, b + p.second};
}

// specialization
template MeshVertexReverseList MeshVertexReverseList::from_mesh(
        size_t, const IndexMat<4>&);
template MeshVertexReverseList MeshVertexReverseList::from_mesh(
        size_t, const IndexMat<3>&);
