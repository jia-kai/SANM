/**
 * \file tests/tensor.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/sparse_solver.h"
#include "libsanm/tensor_svd.h"
#include "tests/helper.h"

#include <cstring>

using namespace sanm;
using namespace test;

TEST_CASE("Tensor.MemoryManagement") {
    TensorND x{TensorShape{2, 3}};
    x.woptr()[0] = 4;
    auto p0 = x.ptr();
    x.set_shape({2});
    REQUIRE(p0[0] == 4);
    x.fill_with_inplace(1.2);
    REQUIRE(p0[0] == 1.2);
    REQUIRE(p0 == x.ptr());
    x.set_shape({20000});
    REQUIRE(p0 != x.ptr());

    x.fill_with_inplace(2.5);
    p0 = x.ptr();
    REQUIRE(!x.empty());
    auto y = x;

    REQUIRE(y.ptr() == p0);
    REQUIRE(x.ptr() == p0);

    x *= 2;

    REQUIRE(y.ptr() == p0);
    REQUIRE(x.ptr() != p0);

    REQUIRE(y.ptr()[0] == 2.5);
    REQUIRE(x.ptr()[0] == 5);
}

TEST_CASE("Tensor.SparseSolver") {
    constexpr size_t N = 8;
    TensorRNG rng;
    TensorND A = rng({N, N}), b = rng({N});
    auto aptr = A.rwptr();
    for (size_t i = 0; i < N; ++i) {
        aptr[i * N + (i + 2) % N] = 0;
    }
    auto make_solver = [&]() {
        auto solver = std::make_unique<SparseSolver>(N);
        auto builder = solver->make_builder();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (fp_t v = aptr[i * N + j]; v) {
                    builder->add_constraint(i, j, v);
                }
            }
        }
        return solver;
    };

    auto dotmv = [](const TensorND& a, const TensorND& x) {
        return TensorND{}.as_mm(a, x.reshape({N, 1})).reshape({N});
    };

    auto solver0 = make_solver();
    solver0->prepare();
    auto x0 = solver0->solve(b);
    require_tensor_eq(dotmv(A, x0), b);

    fp_t alpha = 0.01;
    auto solver1 = make_solver();
    solver1->prepare(alpha);
    auto x1 = solver1->solve(b);
    TensorND aT, aTapI;
    aTapI.as_mm(aT.as_transpose(A), A);
    for (size_t i = 0; i < N; ++i) {
        aTapI.rwptr()[i * N + i] += alpha;
    }
    require_tensor_eq(dotmv(aTapI, x1), dotmv(aT, b));
    REQUIRE(x0.norm_l2() >= x1.norm_l2() + 0.01);
}

TEST_CASE("Tensor.Constant") {
    auto x0 = TensorND{TensorShape{5, 3}}.fill_with_inplace(0),
         x1 = x0.fill_with(1);
    using C = TensorStorage::Constant;

    SECTION("share.0") {
        const fp_t *p0 = x0.ptr(),
                   *p1 = TensorStorage::constant<C::ZERO>()->ptr();
        auto p2 = x0.rwptr();
        REQUIRE(p0 == p1);
        REQUIRE(p0 != p2);
        REQUIRE(p0[0] == 0);
        REQUIRE(p2[0] == 0);
        REQUIRE(x0.ptr() == p2);
    }

    SECTION("share.1") {
        const fp_t *p0 = x1.ptr(),
                   *p1 = TensorStorage::constant<C::ONE>()->ptr();
        auto p2 = x1.rwptr();
        REQUIRE(p0 == p1);
        REQUIRE(p0 != p2);
        REQUIRE(p0[0] == 1);
        REQUIRE(p2[0] == 1);
        REQUIRE(x1.ptr() == p2);
    }

    SECTION("assignment.0") {
        auto y = x0;
        y.set_shape({100, 200}).fill_with_inplace(0);
        REQUIRE(x0.same_storage(y));
        REQUIRE(x0.is_zero());
        REQUIRE_FALSE(x0.is_one());
        REQUIRE(y.is_zero());
        REQUIRE(x0.ptr() == y.ptr());
        REQUIRE(x0.ptr()[0] == 0);

        x0.fill_with_inplace(2.3);
        REQUIRE_FALSE(x0.same_storage(y));
        REQUIRE_FALSE(x0.is_zero());
        REQUIRE(y.is_zero());
        REQUIRE(x0.ptr() != y.ptr());
        REQUIRE(x0.ptr()[0] == 2.3);
        REQUIRE(y.ptr()[0] == 0);

        REQUIRE_THROWS_AS(x0 / y, SANMError);
    }

    SECTION("assignment.1") {
        auto y = x1;
        y.set_shape({100, 200}).fill_with_inplace(1);
        REQUIRE(x1.same_storage(y));
        REQUIRE_FALSE(x1.is_zero());
        REQUIRE(x1.is_one());
        REQUIRE(y.is_one());
        REQUIRE(x1.ptr() == y.ptr());
        REQUIRE(x1.ptr()[0] == 1);

        x1.fill_with_inplace(2.3);
        REQUIRE_FALSE(x1.same_storage(y));
        REQUIRE_FALSE(x1.is_one());
        REQUIRE(y.is_one());
        REQUIRE(x1.ptr() != y.ptr());
        REQUIRE(x1.ptr()[0] == 2.3);
        REQUIRE(y.ptr()[0] == 1);
    }
}

TEST_CASE("Tensor.MatMul") {
    TensorRNG rng;
    auto mm_bruteforce = [](size_t batch, const TensorND& x, const TensorND& y,
                            bool trans_x, bool trans_y) {
        auto apply_trans = [batch](TensorND& dst, const TensorND& src,
                                   bool tr) {
            if (!tr) {
                dst = src;
                return;
            }
            if (batch) {
                dst.as_batched_transpose(src);
            } else {
                dst.as_transpose(src);
            }
        };

        TensorND z, xt, yt;
        apply_trans(xt, x, trans_x);
        apply_trans(yt, y, trans_y);
        const size_t m = xt.shape(xt.rank() - 2), k = xt.shape(xt.rank() - 1),
                     n = yt.shape(yt.rank() - 1);
        if (batch) {
            z.set_shape({batch, m, n});
        } else {
            z.set_shape({m, n});
        }

        auto zptr = z.woptr();
        auto xptr = xt.ptr(), yptr = yt.ptr();
        for (size_t b = 0; b < batch || (!batch && b == 0); ++b) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    fp_t s = 0;
                    for (size_t kk = 0; kk < k; ++kk) {
                        s += xptr[((b * m) + i) * k + kk] *
                             yptr[((b * k) + kk) * n + j];
                    }
                    zptr[((b * m) + i) * n + j] = s;
                }
            }
        }

        return z;
    };

    auto run_with_trans = [&rng, &mm_bruteforce](size_t batch, size_t m,
                                                 size_t k, size_t n,
                                                 bool trans_a, bool trans_b) {
        TensorShape xshp{m, k}, yshp{k, n};
        if (trans_a) {
            std::swap(xshp.dim[0], xshp.dim[1]);
        }
        if (trans_b) {
            std::swap(yshp.dim[0], yshp.dim[1]);
        }
        if (batch) {
            xshp.rank = yshp.rank = 3;
            for (int i : {2, 1}) {
                xshp.dim[i] = xshp.dim[i - 1];
                yshp.dim[i] = yshp.dim[i - 1];
            }
            xshp.dim[0] = yshp.dim[0] = batch;
        }
        TensorND x = rng(xshp), y = rng(yshp), z;
        auto do_mm = [&](bool accum) {
            if (batch) {
                z.as_batched_mm(x, y, accum, trans_a, trans_b);
            } else {
                z.as_mm(x, y, accum, trans_a, trans_b);
            }
        };
        do_mm(false);
        require_tensor_eq(z, mm_bruteforce(batch, x, y, trans_a, trans_b));

        x = rng(xshp);
        y = rng(yshp);
        auto z0 = z;
        do_mm(true);
        require_tensor_eq(z, mm_bruteforce(batch, x, y, trans_a, trans_b) + z0);
    };

    auto run = [&run_with_trans](size_t batch, size_t m, size_t k, size_t n) {
        for (bool tx : {false, true}) {
            for (bool ty : {false, true}) {
                run_with_trans(batch, m, k, n, tx, ty);
            }
        }
    };

    SECTION("square") {
        for (int i : {3, 9}) {
            run(5, i, i, i);
        }
        run(0, 7, 7, 7);
    }

    SECTION("non-square") {
        run(9, 3, 7, 8);
        run(0, 4, 5, 6);
    }
}

TEST_CASE("Tensor.MatInv") {
    TensorRNG rng{1., 2.};
    auto run = [&](size_t batch, size_t dim) {
        auto x = rng({batch, dim, dim}), y = TensorND{}.as_batched_matinv(x),
             z = TensorND{}.as_batched_mm(x, y);
        auto pz = z.ptr();
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    INFO(ssprintf("batch=%zu loc=%zu,%zu", i, j, k));
                    REQUIRE(pz[(i * dim + j) * dim + k] ==
                            Approx(j == k).margin(1e-5));
                }
            }
        }
    };
    for (int i = 6; i <= 10; ++i) {
        run(i, i - 5);
    }
}

TEST_CASE("Tensor.SimpleArith") {
    TensorRNG rng;
    TensorND a = rng({3, 5}), b = rng({3, 5});
    const size_t tot = a.shape().total_nr_elems();

    SECTION("plus") {
        TensorND c = a + b;
        for (size_t i = 0; i < tot; ++i) {
            REQUIRE(c.ptr()[i] == Approx(a.ptr()[i] + b.ptr()[i]));
        }
    }

    SECTION("mul") {
        TensorND c = a * b;
        for (size_t i = 0; i < tot; ++i) {
            REQUIRE(c.ptr()[i] == Approx(a.ptr()[i] * b.ptr()[i]));
        }
    }

    SECTION("sub-assign") {
        TensorND c = a - b;
        auto p0 = a.ptr();
        a -= b;
        REQUIRE(a.ptr() == p0);
        require_tensor_eq(c, a);
    }

    SECTION("div-assign") {
        TensorND c = a / b;
        auto p0 = a.ptr();
        a /= b;
        REQUIRE(a.ptr() == p0);
        require_tensor_eq(c, a);
    }
}

TEST_CASE("Tensor.SimpleArithBcast") {
    TensorRNG rng;
    TensorND a = rng({3, 5, 7});

    SECTION("left") {
        TensorND b = rng({1, 5, 7}), c0 = a - b, c1 = b - a;
        REQUIRE(a.shape() == c0.shape());
        REQUIRE(a.shape() == c1.shape());
        auto aptr = a.ptr(), bptr = b.ptr(), c0ptr = c0.ptr(), c1ptr = c1.ptr();
        for (int i = 0, p = 0; i < 3; ++i) {
            for (int j = 0; j < 35; ++j, ++p) {
                fp_t ai = aptr[p], bi = bptr[j];
                REQUIRE(ai - bi == c0ptr[p]);
                REQUIRE(bi - ai == c1ptr[p]);
            }
        }
    }

    SECTION("mid") {
        TensorND b = rng({3, 1, 7}), c0 = a - b, c1 = b - a;
        REQUIRE(a.shape() == c0.shape());
        REQUIRE(a.shape() == c1.shape());
        auto aptr = a.ptr(), bptr = b.ptr(), c0ptr = c0.ptr(), c1ptr = c1.ptr();
        for (int i = 0, p = 0; i < 3; ++i) {
            for (int j = 0; j < 5; ++j) {
                for (int k = 0; k < 7; ++k, ++p) {
                    fp_t ai = aptr[p], bi = bptr[i * 7 + k];
                    REQUIRE(ai - bi == c0ptr[p]);
                    REQUIRE(bi - ai == c1ptr[p]);
                }
            }
        }
    }

    SECTION("right") {
        TensorND b = rng({3, 5, 1}), c0 = a - b, c1 = b - a;
        REQUIRE(a.shape() == c0.shape());
        REQUIRE(a.shape() == c1.shape());
        auto aptr = a.ptr(), bptr = b.ptr(), c0ptr = c0.ptr(), c1ptr = c1.ptr();
        for (int i = 0, p = 0; i < 15; ++i) {
            for (int j = 0; j < 7; ++j, ++p) {
                fp_t ai = aptr[p], bi = bptr[i];
                REQUIRE(ai - bi == c0ptr[p]);
                REQUIRE(bi - ai == c1ptr[p]);
            }
        }
    }

    SECTION("scalar") {
        TensorND b = rng({3}), c0 = a - b, c1 = b - a;
        REQUIRE(a.shape() == c0.shape());
        REQUIRE(a.shape() == c1.shape());
        auto aptr = a.ptr(), bptr = b.ptr(), c0ptr = c0.ptr(), c1ptr = c1.ptr();
        for (int i = 0, p = 0; i < 3; ++i) {
            for (int j = 0; j < 35; ++j, ++p) {
                fp_t ai = aptr[p], bi = bptr[i];
                REQUIRE(ai - bi == c0ptr[p]);
                REQUIRE(bi - ai == c1ptr[p]);
            }
        }
    }

    SECTION("accum_bcast") {
        TensorND a = rng({3, 1}), b = rng({3, 5}), c = b;
        c.accum_mul(a, 4.2);
        auto cptr = c.ptr(), aptr = a.ptr(), bptr = b.ptr();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 5; ++j) {
                REQUIRE(cptr[i * 5 + j] ==
                        Approx(bptr[i * 5 + j] + aptr[i] * 4.2));
            }
        }
    }
}

TEST_CASE("Tensor.MatMulVecItemLeft") {
    TensorRNG rng;
    auto run = [&rng](size_t B, size_t M, size_t K, size_t P, size_t N) {
        TensorND x = rng({B, M * K, P}), y = rng({B, K, N}), z;
        z.as_batched_mm_vecitem_left(x, y);
        REQUIRE(z.shape() == TensorShape({B, M * N, P}));
        for (size_t b = 0; b < B; ++b) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    for (size_t p = 0; p < P; ++p) {
                        fp_t sum = 0;
                        for (size_t k = 0; k < K; ++k) {
                            sum += x.ptr()[((b * M + m) * K + k) * P + p] *
                                   y.ptr()[(b * K + k) * N + n];
                        }
                        REQUIRE(z.ptr()[((b * M + m) * N + n) * P + p] ==
                                Approx(sum));
                    }
                }
            }
        }
    };
    SECTION("run0") { run(2, 3, 4, 5, 6); }
    SECTION("run1") { run(7, 8, 9, 10, 11); }
}

TEST_CASE("Tensor.DetCofactor") {
    TensorRNG rng{1, 3};

    SECTION("det-inv") {
        for (size_t s : {1, 2, 3, 4, 20}) {
            auto x0 = rng({10, s, s}), x0inv = x0.batched_matinv(),
                 det0 = x0.batched_determinant(),
                 det1 = x0inv.batched_determinant();

            REQUIRE(det0.shape() == TensorShape({10, 1}));

            for (size_t i = 0; i < 10; ++i) {
                fp_t v0 = det0.ptr()[i], v1 = det1.ptr()[i];
                REQUIRE(v0 == Approx(1 / v1));
            }
        }
    }

    auto compute_cofactor = [](const TensorND& inp) -> TensorND {
        const size_t batch = inp.shape(0), dim = inp.shape(1);
        auto ret = inp.make_same_shape();
        auto iptr = inp.ptr();
        auto optr = ret.woptr();

        TensorND tmp{TensorShape{1, dim - 1, dim - 1}}, tmp_det;
        auto tmp_ptr = tmp.woptr();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    for (size_t i1 = 0; i1 < dim; ++i1) {
                        if (i1 == i) {
                            continue;
                        }
                        for (size_t j1 = 0; j1 < dim; ++j1) {
                            if (j1 == j) {
                                continue;
                            }
                            tmp_ptr[(i1 - (i1 > i)) * (dim - 1) +
                                    (j1 - (j1 > j))] =
                                    iptr[(b * dim + i1) * dim + j1];
                        }
                    }
                    fp_t v = tmp_det.as_batched_determinant(tmp).ptr()[0];
                    if ((i + j) % 2) {
                        v = -v;
                    }
                    optr[(b * dim + i) * dim + j] = v;
                }
            }
        }
        return ret;
    };

    auto check_cofactor = [&](const TensorND& x) {
        auto xco = x.batched_cofactor(), xco_chk = compute_cofactor(x);
        require_tensor_eq(xco, xco_chk, std::make_pair<fp_t>(1e-4, 1e-4));
    };

    SECTION("cofactor") {
        for (size_t s : {2, 3, 4, 5, 12}) {
            auto x = rng({10, s, s});
            check_cofactor(x);
            auto xptr = x.rwptr();

            // copy one row to another
            for (size_t i = 0; i < 10; ++i) {
                memcpy(xptr + (i * s) * s, xptr + ((i * s) + 1) * s,
                       sizeof(fp_t) * s);
            }

            check_cofactor(x);

            if (s >= 3) {
                // copy one row to another
                for (size_t i = 0; i < 10; ++i) {
                    memcpy(xptr + ((i * s) + 2) * s, xptr + ((i * s) + 1) * s,
                           sizeof(fp_t) * s);
                }
                check_cofactor(x);
            }
        }
    }
}

TEST_CASE("Tensor.PolyMat") {
    TensorRNG rng;
    auto eval_poly = [](const TensorArray& a, fp_t x) {
        TensorND sum;
        for (auto iter = a.rbegin(); iter != a.rend(); ++iter) {
            if (sum.empty()) {
                sum = *iter;
            } else {
                sum = sum * x + *iter;
            }
        }
        return sum;
    };
    auto run = [&](size_t batch, size_t mdim, size_t nr_term) {
        TensorArray coeffs(nr_term);
        for (size_t i = 0; i < nr_term; ++i) {
            coeffs[i] = rng({batch, mdim, mdim});
        }
        size_t det_nr_term = (nr_term - 1) * mdim + 1;
        TensorArray det_coeffs(det_nr_term);
        for (size_t i = 0; i < det_nr_term; ++i) {
            det_coeffs[i] = compute_polymat_det_coeff(coeffs, i);
        }

        auto chk_xs = rng({det_nr_term});
        for (size_t i = 0; i < det_nr_term; ++i) {
            fp_t x = chk_xs.ptr()[i];
            TensorND det_expect = eval_poly(coeffs, x).batched_determinant(),
                     det_get = eval_poly(det_coeffs, x);
            require_tensor_eq(det_expect, det_get);
        }
    };

    SECTION("c2.1") { run(1, 2, 1); }
    SECTION("c2.2") { run(1, 2, 2); }
    SECTION("c3.1") { run(1, 3, 3); }
    SECTION("c3.2") { run(4, 3, 5); }
    SECTION("c4") { run(2, 4, 3); }
    SECTION("c5") { run(3, 5, 4); }
    SECTION("c6") { run(3, 6, 9); }
    SECTION("c7") { run(5, 7, 8); }
}

TEST_CASE("Tensor.ReduceSum") {
    TensorRNG rng;

    auto run = [](const TensorND& src, const TensorShape& dshape) {
        sanm_assert(src.rank() == 3 && dshape.rank == 3);
        TensorND ret{dshape};
        ret.fill_with_inplace(0);
        size_t sx = src.shape(0), sy = src.shape(1), sz = src.shape(2),
               dx = dshape[1] * dshape[2], dy = dshape[2], dz = 1;
        if (dshape[0] == 1) {
            dx = 0;
        }
        if (dshape[1] == 1) {
            dy = 0;
        }
        if (dshape[2] == 1) {
            dz = 0;
        }
        auto dptr = ret.rwptr();
        auto sptr = src.ptr();
        for (size_t i = 0, is = 0; i < sx; ++i) {
            for (size_t j = 0; j < sy; ++j) {
                for (size_t k = 0; k < sz; ++k, ++is) {
                    dptr[i * dx + j * dy + k * dz] += sptr[is];
                }
            }
        }

        return ret;
    };

    SECTION("left") {
        auto x = rng({6, 7, 8}), y0 = x.reduce_sum(0, true),
             y1 = run(x, y0.shape());
        require_tensor_eq(y0, y1);
    }

    SECTION("mid") {
        auto x = rng({6, 7, 8}), y0 = x.reduce_sum(1, false),
             y1 = run(x, {6, 1, 8}).reshape({6, 8});
        require_tensor_eq(y0, y1);
    }

    SECTION("right") {
        auto x = rng({6, 7, 8}), y0 = x.reduce_sum(2, true),
             y1 = run(x, y0.shape());
        require_tensor_eq(y0, y1);
    }

    SECTION("flatten") {
        auto x = rng({6, 7, 8}), y0 = x.reduce_sum(-1, true),
             y1 = run(x, {6, 1, 1}).reshape({6, 1});
        require_tensor_eq(y0, y1);
    }
}

TEST_CASE("Tensor.Broadcast") {
    TensorRNG rng;
    auto run = [&](size_t s0, size_t s1, size_t s2) {
        for (int axis = 0; axis < 3; ++axis) {
            TensorShape dshape{s0, s1, s2}, sshape = dshape;
            sshape.dim[axis] = 1;
            TensorND src = rng(sshape), dst;
            dst.as_broadcast(src, axis, dshape[axis]);
            size_t stride[3] = {sshape[1] * sshape[2], sshape[2], 1};
            stride[axis] = 0;

            auto dptr = dst.ptr(), sptr = src.ptr();

            for (size_t i = 0; i < s0; ++i) {
                for (size_t j = 0; j < s1; ++j) {
                    for (size_t k = 0; k < s2; ++k) {
                        REQUIRE(dptr[(i * s1 + j) * s2 + k] ==
                                sptr[i * stride[0] + j * stride[1] +
                                     k * stride[2]]);
                    }
                }
            }
        }
    };

    SECTION("small") { run(2, 3, 4); }
    SECTION("large") { run(13, 11, 9); }
}

TEST_CASE("Tensor.Transpose") {
    TensorRNG rng;
    auto run = [&](size_t batch, size_t m, size_t n) {
        auto x = rng({batch, m, n}), y = x.batched_transpose();
        auto xptr = x.ptr(), yptr = y.ptr();
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < m; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    REQUIRE(xptr[((i * m) + j) * n + k] ==
                            yptr[((i * n) + k) * m + j]);
                }
            }
        }
    };

    for (size_t i = 1; i < 6; ++i) {
        for (size_t j = 1; j < 6; ++j) {
            run(4, i, j);
        }
    }
}

TEST_CASE("Tensor.LinearTransform") {
    TensorRNG rng;
    auto make = [&](size_t batch, size_t m, size_t n,
                    bool full) -> StSparseLinearTrans {
        TensorShape shp;
        if (batch) {
            shp = {batch, m, n};
        } else {
            shp = {m, n};
        }
        if (full) {
            return {StSparseLinearTrans::FULL, batch != 0, rng(shp)};
        } else {
            sanm_assert(m == n);
            --shp.rank;
            return {StSparseLinearTrans::ELEMWISE, batch != 0, rng(shp)};
        }
    };
    auto run = [&](size_t batch, size_t m, size_t k, size_t n, bool outer_full,
                   bool inner_full) {
        StSparseLinearTrans lt0 = make(batch, m, k, outer_full),
                            lt1 = make(batch, k, n, inner_full),
                            ltcomp =
                                    inner_full
                                            ? lt0.compose_with_full(lt1.coeff())
                                            : lt0.compose_with_elemwise(
                                                      lt1.coeff());
        TensorND x = batch ? rng({batch, n}) : rng({n});
        require_tensor_eq(lt0.apply(lt1.apply(x)), ltcomp.apply(x));
        if (!inner_full) {
            // add full and elemwise
            StSparseLinearTrans lt2 = make(batch, k, n, true), lts;
            lts += lt1;
            require_tensor_eq(lt1.apply(x), lts.apply(x));
            lts += lt2;
            require_tensor_eq(lt1.apply(x) + lt2.apply(x), lts.apply(x));

            lts = {};
            lts += lt2;
            require_tensor_eq(lt2.apply(x), lts.apply(x));
            lts += lt1;
            require_tensor_eq(lt1.apply(x) + lt2.apply(x), lts.apply(x));
        }

        ltcomp = lt1.compose_with_scaling(2.3);
        require_tensor_eq(lt1.apply(x * 2.3), ltcomp.apply(x));
    };

    for (size_t b : {0, 3}) {
        for (bool fo : {false, true}) {
            for (bool fi : {false, true}) {
                run(b, 5, 5, 5, fo, fi);
                if (fo) {
                    run(b, 5, 8, 8, fo, fi);
                }
                if (fi) {
                    run(b, 7, 7, 6, fo, fi);
                }
                if (fo && fi) {
                    run(b, 3, 4, 5, fo, fi);
                }
            }
        }
    }
}

TEST_CASE("Tensor.Pow") {
    TensorRNG rng{.5, 8.};
    TensorND x = rng({5, 3});
    auto xptr = x.ptr();
    for (fp_t exp : {-3., -2., -1., -.5, 0., .5, 1., 2., 3., 2.3}) {
        auto y = x.pow(exp);
        auto yptr = y.ptr();
        for (size_t i = 0, it = x.shape().total_nr_elems(); i < it; ++i) {
            REQUIRE(yptr[i] == Approx(std::pow(xptr[i], exp)));
        }
    }
}

TEST_CASE("Tensor.SVD") {
    auto xty = [](const TensorND& x, const TensorND& y) {
        return x.batched_transpose().batched_mm(y);
    };
    auto xyt = [](const TensorND& x, const TensorND& y) {
        return x.batched_mm(y.batched_transpose());
    };
    auto xtx = [&](const TensorND& x) { return xty(x, x); };
    auto require_sym = [](TensorND& x) {
        x += x.batched_transpose();
        x *= 0.5;
    };
    auto numgrad = [](TensorND& m, const TensorND& u, const TensorND& s,
                      const TensorND& w, TensorND& dudm, TensorND& dsdm,
                      TensorND& dwdm, bool require_rotation) {
        const size_t batch = m.shape(0), n = m.shape(1);
        dudm.set_shape({batch, n * n, n * n});
        dsdm.set_shape({batch, n, n * n});
        dwdm.set_shape({batch, n * n, n * n});
        auto mptr = m.rwptr();
        TensorND u1, w1, s1, u2, w2, s2;

        auto copy_to = [n](TensorND& dst, const TensorND& src, size_t i,
                           size_t j) {
            auto dptr = dst.rwptr();
            auto sptr = src.ptr();
            size_t off = i * n + j;
            size_t kt = src.shape().total_nr_elems();
            sanm_assert(dst.shape().total_nr_elems() == kt * n * n, "%s vs %s",
                        dst.shape().str().c_str(), src.shape().str().c_str());
            for (size_t k = 0; k < kt; ++k) {
                dptr[k * n * n + off] = sptr[k];
            }
        };
        std::unique_ptr<fp_t[]> mval0{new fp_t[batch]};
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                constexpr fp_t eps = 1e-4;
                for (size_t i = 0; i < batch; ++i) {
                    fp_t& mv = mptr[(i * n + j) * n + k];
                    mval0[i] = mv;
                    mv += eps;
                }
                m.compute_batched_svd_w(u1, s1, w1, require_rotation);
                for (size_t i = 0; i < batch; ++i) {
                    mptr[(i * n + j) * n + k] = mval0[i] - eps;
                }
                m.compute_batched_svd_w(u2, s2, w2, require_rotation);
                for (size_t i = 0; i < batch; ++i) {
                    mptr[(i * n + j) * n + k] = mval0[i];
                }
                (u1 -= u2) *= 1 / (eps * 2);
                (s1 -= s2) *= 1 / (eps * 2);
                (w1 -= w2) *= 1 / (eps * 2);
                copy_to(dudm, u1, j, k);
                copy_to(dsdm, s1, j, k);
                copy_to(dwdm, w1, j, k);
            }
        }
    };
    auto make_id_trans = [](size_t batch, size_t size) {
        return StSparseLinearTrans{
                StSparseLinearTrans::ELEMWISE, true,
                TensorND{TensorShape{batch, size}}.fill_with_inplace(1)};
    };
    auto chk = [](const TensorND& x, const TensorND& y, fp_t eps = 1e-5) {
        require_tensor_eq(x, y, std::make_pair(eps, eps), 1e-6);
    };
    TensorRNG rng;
    auto run_with_v = [&](TensorND& m, bool require_rotation,
                          bool check_su = true, bool compute_Uk = true) {
        const size_t batch = m.shape(0), n = m.shape(1);
        TensorND eye;
        eye.as_batched_diag(TensorND{{batch, n}}.fill_with_inplace(1));
        TensorND u, s, w;
        m.compute_batched_svd_w(u, s, w, require_rotation);
        chk(xtx(u), eye);
        chk(xtx(w), eye);
        TensorND vT;
        vT.as_batched_mm(u.batched_transpose(), w);
        TensorND sdiag = TensorND{}.as_batched_diag(s);
        chk(u.batched_mm(sdiag).batched_mm(vT), m);

        {
            auto det = w.batched_determinant();
            auto p = det.ptr();
            size_t nr_pos = 0;
            for (size_t i = 0; i < batch; ++i) {
                nr_pos += p[i] > 0;
            }
            if (require_rotation) {
                REQUIRE(nr_pos == batch);
            } else {
                REQUIRE(nr_pos > 0);
                REQUIRE(nr_pos < batch);
            }
        }

        // test numerical grad
        TensorND num_dudm, num_dsdm, num_dwdm;
        numgrad(m, u, s, w, num_dudm, num_dsdm, num_dwdm, require_rotation);
        StSparseLinearTrans dudm, dsdm, dwdm;
        svd_w_grad_revmode(dudm, u, s, w, make_id_trans(batch, n * n), {}, {});
        svd_w_grad_revmode(dsdm, u, s, w, {}, make_id_trans(batch, n), {});
        svd_w_grad_revmode(dwdm, u, s, w, {}, {}, make_id_trans(batch, n * n));
        if (check_su) {
            chk(dsdm.as_full(), num_dsdm);
            chk(dudm.as_full(), num_dudm);
        }
        chk(dwdm.as_full(), num_dwdm);

        // test bias solving
        TensorND mUk, mSk, mWk,
                mMk = rng({batch, n, n}), mMbiask = rng({batch, n, n}),
                mBu = rng({batch, n, n}), mBw = rng({batch, n, n});
        require_sym(mBu);
        require_sym(mBw);
        svd_w_taylor_fwd(mUk, mSk, mWk, mMk, mMbiask, u, s, w,
                         compute_Uk ? &mBu : nullptr, mBw);
        chk(xty(w, mWk) + xty(mWk, w), -mBw);
        if (compute_Uk) {
            chk(xty(u, mUk) + xty(mUk, u), -mBu);
        }
        if (check_su) {
            chk(mMk, mMbiask + mUk.batched_mm(sdiag).batched_mm(vT) +
                             u.batched_mm(TensorND{}.as_batched_diag(mSk))
                                     .batched_mm(vT) +
                             u.batched_mm(sdiag)
                                     .batched_mm(mUk.batched_transpose())
                                     .batched_mm(w) +
                             u.batched_mm(sdiag)
                                     .batched_mm(u.batched_transpose())
                                     .batched_mm(mWk));
        }
        TensorND v = vT.batched_transpose(),
                 should_sym = (xty(u, mMk - mMbiask) -
                               sdiag.batched_mm(u.batched_transpose())
                                       .batched_mm(mWk))
                                      .batched_mm(v);
        chk(should_sym, should_sym.batched_transpose());

        // test PW bias solving
        TensorND mBm = rng({batch, n, n}), mBp = rng({batch, n, n}),
                 mBpw = rng({batch, n, n}), mPk;
        require_sym(mBm);
        require_sym(mBp);
        svd_w_taylor_fwd_p(mPk, mWk, mMk, u, s, w, mBm, mBp, mBpw);
        auto p = u.batched_mm(sdiag).batched_mm(u.batched_transpose());
        chk(mPk, mPk.batched_transpose());
        chk(xty(mPk, p) + xty(p, mPk) + mBp, xyt(m, mMk) + xyt(mMk, m) + mBm);
        chk(mMk, p.batched_mm(mWk) + mPk.batched_mm(w) + mBpw, 2e-4);
    };
    auto run = [&](size_t batch, size_t n, bool require_rotation) {
        auto m = rng({batch, n, n});
        run_with_v(m, require_rotation);
    };

    SECTION("small") { run(5, 3, false); }
    SECTION("small-rot") { run(5, 3, true); }
    SECTION("big") {
        run(10, 7, false);
        run(7, 9, true);
    }
    SECTION("eq-singular") {
        constexpr size_t batch = 8, n = 4;
        TensorND m = rng({batch, n, n}), u, s, w;
        m.compute_batched_svd_w(u, s, w);
        auto sptr = s.rwptr();
        auto test = [&](bool compute_Uk = true) {
            m = u.batched_mm(TensorND{}.as_batched_diag(s)).batched_mm(w);
            run_with_v(m, false, false, compute_Uk);
            run_with_v(m, true, false, compute_Uk);
        };

        // two identical values: choose the third one
        for (size_t i = 0; i < batch; ++i) {
            sptr[i * n + 1] = sptr[i * n];
        }
        test();

        // three identical values: negate them
        for (size_t i = 0; i < batch; ++i) {
            sptr[i * n + 2] = sptr[i * n + 1] = sptr[i * n];
        }
        test();
        // check the case when no Uk is computed
        run_with_v(m, true, false, false);
    }
}
