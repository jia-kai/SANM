/**
 * \file libsanm/anm.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/anm.h"
#include "libsanm/sparse_solver.h"
#include "libsanm/strio.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

using namespace sanm;

namespace {
class IdentitySparseLinearDesc final : public SparseLinearDesc {
    TensorShape m_inp_shape, m_out_shape;
    mutable std::vector<InputElem> m_tmp_elem;

public:
    IdentitySparseLinearDesc(const TensorShape& inp_shape,
                             const TensorShape& out_shape)
            : m_inp_shape{inp_shape}, m_out_shape{out_shape} {
        init_multi_thread(1);
    }

    TensorShape out_shape() const override { return m_out_shape; }

    TensorShape inp_shape() const override { return m_inp_shape; }

    void init_multi_thread(size_t nr_thread) override {
        if (nr_thread > m_tmp_elem.size()) {
            m_tmp_elem.resize(nr_thread);
            for (auto& i : m_tmp_elem) {
                i.coeff = 1;
            }
        }
    }

    Linear1d get(size_t dst_index, size_t thread_id) const override {
        auto& elem = m_tmp_elem.at(thread_id);
        elem.idx = dst_index;
        return {&elem, 1};
    }
};
}  // anonymous namespace

/* ======================= SparseLinearDesc ======================= */

void SparseLinearDesc::init_multi_thread(size_t) {}

TensorND SparseLinearDesc::apply(const TensorND& x) const {
    sanm_assert(inp_shape() == x.shape(), "shape mismatch: %s vs %s",
                inp_shape().str().c_str(), x.shape().str().c_str());
    TensorND ret{out_shape()};
    if (x.is_zero()) {
        return ret.fill_with_inplace(0);
    }
    const size_t orange = ret.shape().total_nr_elems(),
                 irange = x.shape().total_nr_elems();
    fp_t* optr = ret.woptr();
    const fp_t* iptr = x.ptr();
    for (size_t i = 0; i < orange; ++i) {
        fp_t sum = 0;
        for (InputElem j : this->get(i, 0)) {
            sanm_assert(j.idx < irange);
            sum += iptr[j.idx] * j.coeff;
        }
        optr[i] = sum;
    }
    return ret;
}

SparseLinearDescPtr SparseLinearDesc::make_identity(
        const TensorShape& inp_shape, const Maybe<TensorShape>& out_shape) {
    return std::make_shared<IdentitySparseLinearDesc>(
            inp_shape, out_shape.valid() ? out_shape.val() : inp_shape);
}

SparseLinearDescCompressed::Linear1d SparseLinearDescCompressed::get(
        size_t dst_index, size_t) const {
    auto&& p = m_oidx_input_elem.at(dst_index);
    auto ptr = m_all_input_elem.data();
    return {ptr + p.first, ptr + p.second};
}

/* ======================= ANMDriverHelper ======================= */
ANMDriverHelper::ANMDriverHelper(symbolic::VarNode* f,
                                 SparseLinearDescPtr remap_inp,
                                 SparseLinearDescPtr remap_out,
                                 const TensorShape& x_shape,
                                 const HyperParam& hyper_param)
        : m_hyper_param{hyper_param},
          m_func_def{f},
          m_remap_inp{std::move(remap_inp)},
          m_remap_out{std::move(remap_out)},
          m_max_a_bound{unary_polynomial::stable_x_range(hyper_param.order)},
          m_x_shape{x_shape},
          m_nr_unknown{x_shape.total_nr_elems()} {
    sanm_assert(hyper_param.order >= 2 && m_remap_inp && m_remap_out,
                "order=%d remap_inp=%p remap_out=%p", hyper_param.order,
                m_remap_inp.get(), m_remap_out.get());
}

void ANMDriverHelper::init_xt0(const TensorND& x, fp_t t) {
    auto xflat = x.flatten_as_vec();
    size_t nr = xflat.shape(0);
    sanm_assert(nr == m_nr_unknown);
    m_xt0.set_shape({nr + 1});
    m_xt0.copy_from_sub_batch(xflat, 0, 0, nr);
    m_xt0.rwptr()[nr] = t;
}

void ANMDriverHelper::estimate_valid_range() {
    fp_t x1 = m_xt_coeffs[1].norm_l2(),
         xback = std::max<fp_t>(m_xt_coeffs.back().norm_l2(), 1e-15);
    fp_t a_bound = std::pow(m_hyper_param.maxr / xback * x1,
                            fp_t(1) / fp_t(m_hyper_param.order - 1));
    a_bound = std::min(a_bound, m_max_a_bound);
    m_t_coeffs.resize(m_xt_coeffs.size());
    for (size_t i = 0; i < m_xt_coeffs.size(); ++i) {
        m_t_coeffs[i] = m_xt_coeffs[i].ptr()[m_nr_unknown];
    }
    sanm_assert(m_t_coeffs[1] > 0);
#if 1
    m_t_max_a = a_bound;
    m_t_max = unary_polynomial::eval(m_t_coeffs, a_bound);
#else
    // global optimization in brent seems wrong
    std::tie(m_t_max_a, m_t_max) =
            unary_polynomial::maximize(m_t_coeffs, -a_bound, a_bound);
#endif
    sanm_assert(m_t_max > m_t_coeffs[0],
                "t does not incr at iter %zu: t0=%g tmax=%g bound=%g "
                "coeffs(neg)=%s",
                m_iter, m_t_coeffs[0], m_t_max, a_bound,
                str(m_t_coeffs).c_str());

    static bool use_pade = getenv("SANM_PADE") != nullptr;

    if ((m_hyper_param.use_pade || use_pade) && a_bound < m_max_a_bound) {
        m_pade.init(m_xt_coeffs, !m_hyper_param.xcoeff_l2_penalty, false);
        if (m_pade->estimate_valid_range(a_bound, m_hyper_param.maxr,
                                         m_max_a_bound)) {
            m_t_max_a = m_pade->get_t_max_a();
            m_t_max = m_pade->get_t_max();
        } else {
            m_pade.reset();
        }
    }
}

void ANMDriverHelper::update_approx() {
    m_xt0 = eval_xt(m_t_max_a);
    solve_expansion_coeffs();
}

std::pair<TensorND, fp_t> ANMDriverHelper::eval(fp_t a) const {
    TensorND xt = eval_xt(a);
    return {xt.take_sub(m_x_shape), xt.ptr()[m_nr_unknown]};
}

TensorND ANMDriverHelper::eval_xt(fp_t a) const {
    if (m_pade.valid()) {
        return m_pade->eval_xt(a);
    } else {
        return unary_polynomial::eval_tensor(m_xt_coeffs, a);
    }
}

fp_t ANMDriverHelper::solve_a(fp_t t) const {
    if (t == m_t_max) {
        return m_t_max_a;
    }
    if (m_pade.valid()) {
        return m_pade->solve_a(t);
    }
    sanm_assert(t >= m_t_coeffs[0] && t < m_t_max);
    fp_t l, r;
    if (m_t_max_a > 0) {
        l = 0;
        r = m_t_max_a;
    } else {
        l = -m_t_max_a;
        r = 0;
    }
    return unary_polynomial::solve_eqn(m_t_coeffs, l, r, t);
}

void ANMDriverHelper::solve_expansion_coeffs() {
    SANM_SCOPED_PROFILER("solve_expansion_coeffs");
    m_xt_coeffs.resize(m_hyper_param.order + 1);
    m_xt_coeffs[0] = m_xt0;
    m_t_coeffs.resize(1);
    m_t_coeffs[0] = m_xt0.ptr()[m_nr_unknown];

    bool verbose = verbose_mode();
    if (verbose) {
        printf("=== ANM iter %zu:\n", m_iter);
    }

    symbolic::ParallelTaylorCoeffProp taylor_coeff_prop{m_func_def};
    {
        auto fx0 = m_remap_out->apply(
                taylor_coeff_prop
                        .push_xi(m_remap_inp->apply(prepare_inp(m_xt0)))
                        .gather_yi());
        if (!on_fx0_computed(fx0)) {
            m_xt_coeffs.resize(1);
            return;
        }
    }

    SparseSolver sparse_solver{m_nr_unknown};

    TensorND xgt, x1;
    fp_t xgt_dot_x1 = 0, t1 = 0;
    TensorND grad_t;

    for (int i = 1; i <= m_hyper_param.order; ++i) {
        TensorND bi =
                m_remap_out->apply(taylor_coeff_prop.compute_next_order_bias())
                        .flatten_as_vec();

        // A*x + t*gt + bi = 0 (where A=dy/dx, gt=dy/dt, and bi is taylor bias)
        // A*xgt = gt
        // A*xbi = bi
        // x = -t * xgt - xbi

        TensorND xbi;
        fp_t ti;
        if (i == 1) {
            sanm_assert(bi.is_zero());
            m_remap_inp->init_multi_thread(taylor_coeff_prop.nr_worker());
            m_remap_out->init_multi_thread(taylor_coeff_prop.nr_worker());
            build_sparse_coeff(sparse_solver, taylor_coeff_prop);
            grad_t = get_grad_t().flatten_as_vec();
            sparse_solver.prepare(m_hyper_param.xcoeff_l2_penalty);
            xgt = sparse_solver.solve(grad_t);

            // x1.dot(x1) + t1*t1 = 0
            xbi = bi;
            t1 = ti = 1 / std::sqrt(xgt.squared_norm_l2() + 1);
            if (verbose) {
                printf("gt=%g xgt=%g jacob=%g", grad_t.norm_l2(), xgt.norm_l2(),
                       sparse_solver.coeff_l2());
            }
        } else {
            // xi.dot(x1) + ti*t1 = 0
            xbi = sparse_solver.solve(bi);
            ti = xbi.flat_dot(x1) / (t1 - xgt_dot_x1);
        }

        if (verbose) {
            printf(" %d:(bi=%g xbi=%g)", i, bi.norm_l2(), xbi.norm_l2());
        }

        m_xt_coeffs[i]
                .set_shape({m_nr_unknown + 1})
                .copy_from_sub_batch((xgt * (-ti)) -= xbi, 0, 0, m_nr_unknown);
        m_xt_coeffs[i].rwptr()[m_nr_unknown] = ti;

        if (i == 1) {
            x1 = m_xt_coeffs[i].take_sub({m_nr_unknown});
            xgt_dot_x1 = x1.flat_dot(xgt);
        }

        if (!m_hyper_param.xcoeff_l2_penalty && m_hyper_param.sanity_check) {
            SANM_SCOPED_PROFILER("anm_sanity_check");
            TensorND Ax = sparse_solver.apply(
                             m_xt_coeffs[i].take_sub({m_nr_unknown})),
                     Ax_r = ((grad_t * m_xt_coeffs[i].ptr()[m_nr_unknown]) +=
                             bi);
            Ax_r.inplace_neg().reshape_inplace(Ax.shape());
            Ax.assert_allclose("ANM check coeff eqn", Ax_r);
            fp_t xdot = m_xt_coeffs[1].flat_dot(m_xt_coeffs[i]);
            if (i == 1) {
                sanm_assert(std::fabs(xdot - 1) < 1e-4, "xdot=%g", xdot);
            } else {
                sanm_assert(std::fabs(xdot) < 1e-4, "i=%d: xdot=%g", i, xdot);
            }
        }

        if (i < m_hyper_param.order) {
            taylor_coeff_prop.push_xi(
                    m_remap_inp->apply(prepare_inp(m_xt_coeffs[i])));
        }
    }

    estimate_valid_range();

    if (verbose) {
        printf("\nbound=%g t=%g\n", m_t_max_a, m_t_max);
        printf("x(a):");
        for (auto& i : m_xt_coeffs) {
            printf(" %.3g", i.norm_l2());
        }
        printf("\nt(a):");
        for (fp_t i : m_t_coeffs) {
            printf(" %.3g,", i);
        }
        printf("\n");
        if (m_hyper_param.xcoeff_l2_penalty) {
            printf("xcoeff_l2_penalty=%g\n", m_hyper_param.xcoeff_l2_penalty);
        }
    }

    ++m_iter;
}

bool ANMDriverHelper::verbose_mode() {
    static bool ret = getenv("SANM_VERBOSE") != nullptr;
    return ret;
}

/* ======================= ANMSolverVecScale ======================= */

ANMSolverVecScale::ANMSolverVecScale(symbolic::VarNode* f,
                                     SparseLinearDescPtr remap_inp,
                                     SparseLinearDescPtr remap_out, TensorND x0,
                                     fp_t t0, TensorND v,
                                     const HyperParam& hyper_param)
        : ANMSolverVecScale(f, std::move(remap_inp), std::move(remap_out),
                            x0.shape(), hyper_param) {
    m_v = std::move(v);
    sanm_assert(m_remap_inp->inp_shape() == m_x_shape,
                "linear map expects %s, got x0 %s",
                m_remap_inp->inp_shape().str().c_str(),
                m_x_shape.str().c_str());
    sanm_assert(m_x_shape.total_nr_elems() == m_v.shape().total_nr_elems(),
                "currently we assume the system is a full-rank mapping");
    sanm_assert(m_remap_out->out_shape() == m_v.shape(),
                "output shapes mismatch: %s vs %s",
                m_remap_out->out_shape().str().c_str(),
                m_v.shape().str().c_str());
    init_xt0(x0, t0);
    solve_expansion_coeffs();
}

void ANMSolverVecScale::check_t0v_match(const TensorND& fx) const {
    sanm_assert(fx.shape() == m_v.shape(),
                "fx.shape=%s and v.shape=%s mismatch", fx.shape().str().c_str(),
                m_v.shape().str().c_str());
    auto pfx = fx.ptr(), pv = m_v.ptr();
    size_t nr = fx.shape().total_nr_elems();
    for (size_t i = 0; i < nr; ++i) {
        auto a = pfx[i], b = pv[i] * get_t0();
        auto maxerr = std::max<fp_t>(std::min(std::fabs(a), std::fabs(b)), 1) *
                      m_hyper_param.solution_check_tol;
        if (std::fabs(a + b) > maxerr) {
            throw SANMNumericalError{
                    ssprintf("f(x0)+t0*v is not zero: lhs=%g rhs=%g idx=%zu "
                             "shape=%s iter=%zu",
                             a, b, i, fx.shape().str().c_str(), m_iter)};
        }
    }
}

void ANMSolverVecScale::build_sparse_coeff(
        SparseSolver& sparse_solver,
        symbolic::ParallelTaylorCoeffProp& coeff_prop) const {
    SANM_SCOPED_PROFILER("build_sparse_coeff");

    // equation: map_out(lt_k @ map_in(x) + lt_b) + m_v = 0

    const size_t nr_elem_out = m_remap_out->out_shape().total_nr_elems(),
                 nr_shard = coeff_prop.nr_worker();
    StSparseLinearTrans lt_k;
    std::mutex mutex;

    auto worker = [=, self = static_cast<const ANMSolverVecScale*>(this),
                   &sparse_solver, &coeff_prop, &mutex,
                   &lt_k](size_t shard_id) {
        const size_t begin = shard_id * nr_elem_out / nr_shard,
                     end = (shard_id + 1) * nr_elem_out / nr_shard;
        SparseSolver::SparseMatBuilder* mat_builder = nullptr;

        const fp_t* lt_k_ptr = nullptr;
        size_t lt_odim = 0, lt_idim = 0;
        StSparseLinearTrans::Type lt_type = StSparseLinearTrans::INVALID;
        {
            std::lock_guard<std::mutex> lock{mutex};
            mat_builder = sparse_solver.make_builder(begin);
            if (!lt_k.valid()) {
                lt_k = coeff_prop.get_jacobian();
            }
        }
        lt_odim = lt_k.out_dim();
        lt_idim = lt_k.inp_dim();
        lt_k_ptr = lt_k.coeff().ptr();
        lt_type = lt_k.type();

        auto chk = self->m_remap_inp->out_shape().flatten_batched();
        sanm_assert(lt_k.batch() == chk[0]);
        sanm_assert(lt_idim == chk[1]);
        chk = self->m_remap_out->inp_shape().flatten_batched();
        sanm_assert(lt_k.batch() == chk[0]);
        sanm_assert(lt_odim == chk[1]);

        // called when (batch, oidx) of (lt_k @ map_in(x)) has a term with given
        // coeff on map_in(x)
        auto handle_omap = [self, mat_builder, iidx_mul = lt_idim, shard_id](
                                   size_t i_out, size_t batch, size_t oidx,
                                   fp_t coeff, size_t iidx) {
            sanm_assert(iidx < iidx_mul);
            auto trans =
                    self->m_remap_inp->get(batch * iidx_mul + iidx, shard_id);
            for (auto i : trans) {
                mat_builder->add_constraint(i_out, i.idx, coeff * i.coeff);
            }
        };

        for (size_t i_out = begin; i_out < end; ++i_out) {
            auto trans_from_func = self->m_remap_out->get(i_out, shard_id);
            for (auto func_out_item : trans_from_func) {
                size_t batch = func_out_item.idx / lt_odim,
                       oidx = func_out_item.idx % lt_odim;
                if (lt_type == StSparseLinearTrans::ELEMWISE) {
                    handle_omap(i_out, batch, oidx,
                                lt_k_ptr[batch * lt_odim + oidx] *
                                        func_out_item.coeff,
                                oidx);
                } else {
                    sanm_assert(lt_type == StSparseLinearTrans::FULL);
                    auto kbase = lt_k_ptr + (batch * lt_odim + oidx) * lt_idim;
                    for (size_t iidx = 0; iidx < lt_idim; ++iidx) {
                        handle_omap(i_out, batch, oidx,
                                    kbase[iidx] * func_out_item.coeff, iidx);
                    }
                }
            }
        }
    };
    coeff_prop.run_on_workers(worker);
}

bool ANMSolverVecScale::on_fx0_computed(const TensorND& fx) {
    check_t0v_match(fx);
    return true;
}

/* ======================= ANMEqnSolver ======================= */
ANMEqnSolver::ANMEqnSolver(symbolic::VarNode* f, SparseLinearDescPtr remap_inp,
                           SparseLinearDescPtr remap_out, TensorND x0,
                           TensorND y, const HyperParam& hyper_param)
        : ANMSolverVecScale(f, std::move(remap_inp), std::move(remap_out),
                            x0.shape(), hyper_param),
          m_converge_rms{hyper_param.converge_rms} {
    // we always assume the form f(x) - f(x0) + t * (y + f(x0)) = 0, t from 0
    init_xt0(x0, 0);
    m_eqn_y = std::move(y);
    sanm_assert(x0.shape().total_nr_elems() == m_eqn_y.shape().total_nr_elems(),
                "currently we assume the system is a full-rank mapping");
    sanm_assert(m_remap_out->out_shape() == m_eqn_y.shape(),
                "output shapes mismatch: %s vs %s",
                m_remap_out->out_shape().str().c_str(),
                m_eqn_y.shape().str().c_str());
    solve_expansion_coeffs();
}

ANMEqnSolver& ANMEqnSolver::next_iter() {
    if (m_converged) {
        return *this;
    }
    fp_t a;
    if (get_t_upper() >= 1) {
        a = solve_a(1);
    } else {
        a = get_t_max_a();
    }
    m_xt0 = eval_xt(a);
    m_xt0.rwptr()[m_nr_unknown] = 0;  // set t0 to 0
    solve_expansion_coeffs();
    return *this;
}

bool ANMEqnSolver::on_fx0_computed(const TensorND& fx) {
    if (m_converged) {
        return false;
    }
    m_v = fx + m_eqn_y;
    m_residual_rms = m_v.norm_rms();
    if (m_residual_rms < m_converge_rms) {
        m_converged = true;
        return false;
    }
    return true;
}

/* ======================= ANMImplicitSolver ======================= */
ANMImplicitSolver::ANMImplicitSolver(symbolic::VarNode* f,
                                     SparseLinearDescPtr remap_inp,
                                     SparseLinearDescPtr remap_out,
                                     const TensorND& x0, fp_t t0,
                                     const HyperParam& hyper_param)
        : ANMDriverHelper(f, std::move(remap_inp), std::move(remap_out),
                          x0.shape(), hyper_param) {
    sanm_assert(m_remap_inp->inp_shape().rank == 1 &&
                m_remap_out->out_shape().rank == 1 &&
                m_remap_inp->inp_shape()[0] == m_remap_out->out_shape()[0] + 1);

    sanm_assert(x0.shape() == m_remap_out->out_shape());
    init_xt0(x0, t0);
    solve_expansion_coeffs();
}

bool ANMImplicitSolver::on_fx0_computed(const TensorND& fx) {
    if (m_fx0.empty()) {
        m_fx0 = fx;
    } else {
        m_fx0.assert_allclose("check f(x0, t0)=f(x, t)", fx,
                              m_hyper_param.solution_check_tol);
    }
    return true;
}

void ANMImplicitSolver::build_sparse_coeff(
        SparseSolver& sparse_solver,
        symbolic::ParallelTaylorCoeffProp& coeff_prop) const {
    SANM_SCOPED_PROFILER("build_sparse_coeff");

    // equation: map_out(lt_k @ map_in([x; t]) + lt_b) = 0

    const size_t nr_elem_out = m_remap_out->out_shape().total_nr_elems(),
                 nr_shard = coeff_prop.nr_worker();
    StSparseLinearTrans lt_k;
    std::mutex mutex;

    auto worker = [=, this, &sparse_solver, &coeff_prop, &mutex,
                   &lt_k](size_t shard_id) {
        const size_t begin = shard_id * nr_elem_out / nr_shard,
                     end = (shard_id + 1) * nr_elem_out / nr_shard;
        SparseSolver::SparseMatBuilder* mat_builder = nullptr;

        TensorND& grad_t = m_grad_t;
        const fp_t* lt_k_ptr = nullptr;
        size_t lt_odim = 0, lt_idim = 0;
        StSparseLinearTrans::Type lt_type = StSparseLinearTrans::INVALID;
        {
            std::lock_guard<std::mutex> lock{mutex};
            mat_builder = sparse_solver.make_builder(begin);
            if (!lt_k.valid()) {
                // data structures shared by all workers
                lt_k = coeff_prop.get_jacobian();
                grad_t.set_shape({m_nr_unknown});
            }
        }
        lt_odim = lt_k.out_dim();
        lt_idim = lt_k.inp_dim();
        lt_k_ptr = lt_k.coeff().ptr();
        lt_type = lt_k.type();

        auto chk = m_remap_inp->out_shape().flatten_batched();
        sanm_assert(lt_k.batch() == chk[0]);
        sanm_assert(lt_idim == chk[1]);
        chk = m_remap_out->inp_shape().flatten_batched();
        sanm_assert(lt_k.batch() == chk[0]);
        sanm_assert(lt_odim == chk[1]);

        fp_t cur_grad_t;

        // called when (batch, oidx) of (lt_k @ map_in(x)) has a term with given
        // coeff on map_in(x)
        auto handle_omap = [mat_builder, iidx_mul = lt_idim, shard_id,
                            gt = &cur_grad_t, this](size_t i_out, size_t batch,
                                                    size_t oidx, fp_t coeff,
                                                    size_t iidx) {
            sanm_assert(iidx < iidx_mul);
            auto trans = m_remap_inp->get(batch * iidx_mul + iidx, shard_id);
            for (auto i : trans) {
                fp_t g = coeff * i.coeff;
                if (i.idx < m_nr_unknown) {
                    mat_builder->add_constraint(i_out, i.idx, g);
                } else {
                    *gt += g;
                }
            }
        };

        fp_t* grad_t_ptr = grad_t.rwptr();
        for (size_t i_out = begin; i_out < end; ++i_out) {
            auto trans_from_func = m_remap_out->get(i_out, shard_id);
            cur_grad_t = 0;
            for (auto func_out_item : trans_from_func) {
                size_t batch = func_out_item.idx / lt_odim,
                       oidx = func_out_item.idx % lt_odim;
                if (lt_type == StSparseLinearTrans::ELEMWISE) {
                    handle_omap(i_out, batch, oidx,
                                lt_k_ptr[batch * lt_odim + oidx] *
                                        func_out_item.coeff,
                                oidx);
                } else {
                    sanm_assert(lt_type == StSparseLinearTrans::FULL);
                    auto kbase = lt_k_ptr + (batch * lt_odim + oidx) * lt_idim;
                    for (size_t iidx = 0; iidx < lt_idim; ++iidx) {
                        handle_omap(i_out, batch, oidx,
                                    kbase[iidx] * func_out_item.coeff, iidx);
                    }
                }
            }
            grad_t_ptr[i_out] = cur_grad_t;
        }
    };
    coeff_prop.run_on_workers(worker);
}

const TensorND& ANMImplicitSolver::get_grad_t() const {
    sanm_assert(
            !m_grad_t.empty(),
            "grad_t not initialized; please call build_sparse_coeff() first");
    return m_grad_t;
}
