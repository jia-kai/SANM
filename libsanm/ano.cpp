/**
 * \file libsanm/ano.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/ano.h"

#include <cmath>
#include <random>

using namespace sanm;
using namespace symbolic;

/* ======================= ANOMinimizer ======================= */
ANOMinimizer::ANOMinimizer(symbolic::VarNode* loss, const TensorValueMap& x0,
                           std::unique_ptr<CoeffSolver> coeff_solver,
                           const HyperParam& hyper_param)
        : m_hyper_param{hyper_param},
          m_coeff_solver{std::move(coeff_solver)},
          m_loss_var{loss},
          m_max_a_bound{unary_polynomial::stable_x_range(hyper_param.order)} {
    init_grad(x0);
}

ANOMinimizer::~ANOMinimizer() = default;

ANOMinimizer::Stat ANOMinimizer::update_approx() {
    ++m_iter;
    Stat ret;
    solve_expansion_coeffs();
    ret.a_bound = estimate_valid_range();
    fp_t t_next;
    std::tie(ret.a_m, t_next) =
            unary_polynomial::minimize(m_t_coeffs, -ret.a_bound, ret.a_bound);
    TensorND x_next = unary_polynomial::eval_tensor(m_x_coeffs, ret.a_m);
    init_grad(unpack_x_coeffs(x_next));
    ret.loss_diff = std::fabs(m_t_coeffs[0] - t_next);
    sanm_assert(ret.loss_diff < m_hyper_param.max_loss_diff,
                "loss_diff too large: approx=%g actual=%g a=%g/%g", t_next,
                m_t_coeffs[0], ret.a_m, ret.a_bound);
    return ret;
}

void ANOMinimizer::solve_expansion_coeffs() {
    sanm_assert(m_x_coeffs.size() == 1 && m_t_coeffs.size() == 1);
    TaylorCoeffProp& taylor_prop = m_taylor_prop.val();
    for (int i = 1; i <= m_hyper_param.order; ++i) {
        const TensorND& b = taylor_prop.compute_next_order_bias();
        sanm_assert(b.shape().is_single_scalar());
        auto xtpair =
                m_coeff_solver->solve(i, *b.ptr(), m_x_coeffs, m_t_coeffs);
        // printf("b[%d]=%g\n", i, *b.ptr());
        m_x_coeffs.emplace_back(xtpair.first);
        m_t_coeffs.emplace_back(xtpair.second);
        if (i < m_hyper_param.order) {
            taylor_prop.push_xi(unpack_x_coeffs(xtpair.first));
        }
    }
}

fp_t ANOMinimizer::estimate_valid_range() const {
    auto get_norm = [this](size_t i) {
        fp_t ti = m_t_coeffs[i], s = m_x_coeffs[i].squared_norm_l2() + ti * ti;
        return std::sqrt(s);
    };
    fp_t xback = std::max<fp_t>(get_norm(m_x_coeffs.size() - 1), 1e-9);
    fp_t bound = std::pow(m_hyper_param.maxr / xback * get_norm(1),
                          fp_t(1) / fp_t(m_hyper_param.order - 1));
    bound = std::min(bound, m_max_a_bound);
#if 0
    printf("== iter %zu: dump coeffs (bound=%g minimizer=%g)\n", m_iter, bound,
           unary_polynomial::minimize(m_t_coeffs, -bound, bound).first);
    for (fp_t i : m_t_coeffs) {
        printf("%g ", i);
    }
    printf("\n");
    printf("== dump RMS\n");
    for (auto& i : m_x_coeffs) {
        printf("%g ",
               std::sqrt(i.squared_norm_l2() / i.shape().total_nr_elems()));
    }
    printf("\n");
    printf("press enter to continue ... ");
    fflush(stdout);
    {
        char* line = nullptr;
        size_t size = 0;
        getline(&line, &size, stdin);
        ::free(line);
    }
#endif
    return bound;
}

void ANOMinimizer::init_grad(const TensorValueMap& x0) {
    TaylorCoeffProp& prop = m_taylor_prop.init(m_loss_var, false);
    const TensorND& loss = prop.push_xi(x0);
    sanm_assert(loss.shape().is_single_scalar(), "loss is not scalar: %s",
                loss.shape().str().c_str());
    m_x_coeffs.clear();
    m_t_coeffs.clear();
    m_t_coeffs.emplace_back(*loss.ptr());

    m_x0_vars.clear();
    size_t total_size = 0;
    for (auto& i : x0) {
        total_size += i.second.shape().total_nr_elems();
        m_x0_vars.emplace_back(i.first, i.second.shape());
    }
    m_grad_flat.set_shape({total_size});
    TensorND x0_flat = m_grad_flat.make_same_shape();

    size_t grad_off = 0;
    for (auto& i : x0) {
        auto& jacobian = prop.get_jacobian(i.first);
        sanm_assert(!jacobian.is_batched() && jacobian.out_dim() == 1);
        size_t cur_size = jacobian.inp_dim();
        m_grad_flat.copy_from_sub_batch(jacobian.coeff().flatten_as_vec(),
                                        grad_off, 0, cur_size);
        x0_flat.copy_from_sub_batch(i.second.flatten_as_vec(), grad_off, 0,
                                    cur_size);
        grad_off += cur_size;
    }
    sanm_assert(grad_off == total_size);
    m_x_coeffs.emplace_back(x0_flat);
    m_coeff_solver->init(m_iter, m_grad_flat);
}

TensorValueMap ANOMinimizer::unpack_x_coeffs(const TensorND& xflat) const {
    sanm_assert(xflat.rank() == 1);
    size_t offset = 0;
    TensorValueMap ret;
    for (auto& i : m_x0_vars) {
        auto size = i.second.total_nr_elems();
        TensorND cur;
        cur.copy_from_sub_batch(xflat, 0, offset, size);
        ret.insert(i.first, std::move(cur.reshape_inplace(i.second)));
        offset += size;
    }
    sanm_assert(offset == xflat.shape(0));
    return ret;
}

/* ======================= ANOMinimizer::CoeffSolver ======================= */

using CoeffSolver = ANOMinimizer::CoeffSolver;

std::pair<TensorND, fp_t> CoeffSolver::solve_with_scale(
        TensorND r, const TensorND& grad, size_t order, fp_t b,
        const TensorArray& xprev, unary_polynomial::coeff_t tprev) {
    // xi.dot(x1) + ti*t1 = 1(i == 1)
    // xi.dot(m_grad) + b = ti
    // xi = ki * xrand

    fp_t ki, ti, xr1;
    fp_t rg = r.flat_dot(grad);
    if (order == 1) {
        sanm_assert(b == 0);
        xr1 = r.squared_norm_l2();
        ki = std::sqrt(1 / (xr1 + rg * rg));
    } else {
        sanm_assert(tprev.size() >= 2);
        xr1 = r.flat_dot(xprev[1]);
        ki = -tprev[1] * b / (tprev[1] * rg + xr1);
    }
    ti = ki * rg + b;
    if (order == 1) {
        sanm_assert(std::fabs(ki * ki * xr1 + ti * ti - 1) < 1e-4);
    } else {
        sanm_assert(std::fabs(ki * xr1 + ti * tprev[1]) < 1e-4);
    }
    return {r *= ki, ti};
}

class CoeffSolver::GradScale final : public CoeffSolver {
    TensorND m_grad;
    fp_t m_g2;        //!< m_grad.dot(m_grad)
    fp_t m_k1, m_t1;  //!< x1 = m_k1 * m_grad
public:
    void init(size_t iter, const TensorND& grad) override {
        m_grad = grad;
        m_g2 = grad.squared_norm_l2();
    }

    std::pair<TensorND, fp_t> solve(size_t order, fp_t b,
                                    const TensorArray& xprev,
                                    unary_polynomial::coeff_t tprev) override {
        // xi.dot(x1) + ti*t1 = 1(i == 1)
        // xi.dot(m_grad) + b = ti
        // xi = ki * m_grad
        constexpr fp_t NORM1 = 1;
        fp_t ki, ti;
        if (order == 1) {
            sanm_assert(b == 0);
            ki = m_k1 = std::sqrt(NORM1 / (m_g2 * m_g2 + m_g2));
            ti = m_t1 = m_k1 * m_g2;
            sanm_assert(std::fabs(m_k1 + m_t1) > 1e-3);
        } else {
            ti = b * m_k1 / (m_t1 + m_k1);
            ki = (ti - b) / m_g2;
        }
        sanm_assert(std::fabs(ki * m_g2 + b - ti) < 1e-4);
        sanm_assert(std::fabs(ki * m_k1 * m_g2 + ti * m_t1 -
                              (order == 1 ? NORM1 : 0)) < 1e-4);
        return {m_grad * ki, ti};
    }
};

class CoeffSolver::Random final : public CoeffSolver {
    TensorND m_grad;
    fp_t m_g2, m_g2_sqrt;
    Xorshift128pRng m_rng;
    std::normal_distribution<fp_t> m_normal_dist;
    std::uniform_real_distribution<fp_t> m_angle_dist;

    void fill_with_normal(TensorND& dst) {
        auto ptr = dst.woptr();
        for (size_t i = 0, it = dst.shape().total_nr_elems(); i < it; ++i) {
            ptr[i] = m_normal_dist(m_rng);
        }
    }

    //! generate a uniform random tensor such that it has the given angle to
    //! m_grad
    TensorND gen_xrand(fp_t angle) {
        int iter = 0;
        TensorND r{m_grad.shape()};
        for (;;) {
            ++iter;
            sanm_assert(iter <= 3);
            fill_with_normal(r);
            // project r into d in {x: x.dot(m_grad)=0}
            // r = k * m_grad + d, d.dot(m_grad) = 0
            fp_t k = r.flat_dot(m_grad) / m_g2;
            TensorND& d = r.accum_mul(m_grad, -k);
            fp_t dnorm = d.norm_l2();
            if (dnorm >= 1e-4) {
                fp_t dnorm_req = m_g2_sqrt * std::tan(angle);
                return (d *= (dnorm_req / dnorm)) += m_grad;
            }
        }
    }

public:
    Random(fp_t max_angle, size_t seed)
            : m_rng{seed}, m_angle_dist{0, max_angle} {
        sanm_assert(max_angle > 0 && max_angle < M_PI / 2 * 0.95);
    }

    void init(size_t iter, const TensorND& grad) override {
        m_grad = grad;
        m_g2 = grad.squared_norm_l2();
        sanm_assert(m_g2 > 1e-6);
        m_g2_sqrt = std::sqrt(m_g2);
    }

    std::pair<TensorND, fp_t> solve(size_t order, fp_t b,
                                    const TensorArray& xprev,
                                    unary_polynomial::coeff_t tprev) override {
        TensorND xrand = order == 1 ? m_grad : gen_xrand(m_angle_dist(m_rng));
        return solve_with_scale(std::move(xrand), m_grad, order, b, xprev,
                                tprev);
    }
};

class CoeffSolver::GDApprox final : public CoeffSolver {
    const fp_t m_mom_smooth;
    TensorND m_mom, m_grad;

public:
    explicit GDApprox(fp_t mom) : m_mom_smooth{mom} {}

    void init(size_t iter, const TensorND& grad) override {
        if (iter == 0) {
            m_mom = grad;
        } else {
            m_mom *= m_mom_smooth;
            m_mom += grad;
        }
        m_grad = grad;
    }

    std::pair<TensorND, fp_t> solve(size_t order, fp_t b,
                                    const TensorArray& xprev,
                                    unary_polynomial::coeff_t tprev) override {
        if (order == 1) {
            sanm_assert(b == 0);
            return {m_mom, m_mom.flat_dot(m_grad)};
            // return solve_with_scale(m_grad, m_grad, 1, b, xprev, tprev);
        }
        return {m_mom.fill_with(0), b};
    }
};

std::unique_ptr<CoeffSolver> CoeffSolver::make_grad_scale() {
    return std::make_unique<GradScale>();
}

std::unique_ptr<CoeffSolver> CoeffSolver::make_random(fp_t max_angle,
                                                      size_t seed) {
    return std::make_unique<Random>(max_angle, seed);
}

std::unique_ptr<CoeffSolver> CoeffSolver::make_gd_approx(fp_t momentum) {
    return std::make_unique<GDApprox>(momentum);
}
