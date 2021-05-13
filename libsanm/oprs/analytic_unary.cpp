/**
 * \file libsanm/oprs/analytic_unary.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/oprs/analytic_unary.h"
#include "libsanm/tensor_impl_helper.h"

/*
 * An older implementation (commented out below) uses Faà di Bruno's formula in
 * terms of Bell polynomials, which needs O(k^3) but handles the general case.
 * See https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula and
 * https://en.wikipedia.org/wiki/Bell_polynomials
 *
 * The current version uses coefficient propagation specialized for each opr.
 */

#if 0
class AnalyticUnaryOprMeta::EvalState final : public VarNodeExeCtx::Userdata {
    /*!
     * m_bell_fac[n-1][k-1] is the Bell polynomial multiplied by factorial:
     * f$B_{n,k}k!/n!\f$ evaluated on existig coefficients
     * Multiply by k! because UnaryAnalyticTrait evaluates f^k/k!
     * Divide by n! because we need the final coefficient as
     * d^n(f(g(x))/dx^n / n!
     */
    std::vector<std::vector<TensorND>> m_bell_fac;
    TensorND m_b00;

    const UnaryAnalyticTrait* const m_trait;

public:
    explicit EvalState(UnaryAnalyticMode mode)
            : m_trait{UnaryAnalyticTrait::from_mode(mode)} {}

    TensorND cached_k;
    TensorArray cached_fkb;  //! f^k(g(0))

    const UnaryAnalyticTrait* trait() const { return m_trait; }

    const TensorND& get_bell_fac(int n, int k, const TensorArray& coeffs) {
        if (n == 0 && k == 0) {
            if (m_b00.empty()) {
                m_b00 = coeffs[0].fill_with(1);
            }
            return m_b00;
        }

        sanm_assert(n > 0 && 1 <= k && k <= n &&
                    coeffs.size() >= static_cast<size_t>(n));
        if (m_bell_fac.size() < static_cast<size_t>(n)) {
            m_bell_fac.reserve(std::max<size_t>(m_bell_fac.capacity() * 2, n));
            m_bell_fac.resize(n);
        }
        {
            auto& arr = m_bell_fac[n - 1];
            if (arr.size() < static_cast<size_t>(k)) {
                arr.reserve(std::max<size_t>(arr.capacity() * 2, k));
                arr.resize(k);
            }
        }
        TensorND& Bnk = m_bell_fac[n - 1][k - 1];
        if (!Bnk.empty()) {
            return Bnk;
        }

        Bnk.set_shape(coeffs[0].shape()).woptr();
        bool initialized = false;
        EigenVec Bnk_eig = as_vector_w(Bnk);
        for (int i = 1; i <= n - k + 1; ++i) {
            // x_i = g^i(x) = c_i * i!
            if (i == n && coeffs.size() == static_cast<size_t>(n)) {
                // g^n is zero when computing current order
                sanm_assert(k == 1);
                continue;
            }
            if (i == n && k > 1) {
                // B(n-i, k-1) = 0 for k > 1
                continue;
            }
            if (k == 1 && i < n) {
                // B(n-i, k-1) = 0 for n - i > 0
                continue;
            }

            const TensorND& ci = coeffs[i];
            const TensorND& bp = get_bell_fac(n - i, k - 1, coeffs);

            if (!initialized) {
                Bnk_eig = as_vector_r(ci).array() * as_vector_r(bp).array() * i;
                initialized = true;
            } else {
                Bnk_eig.array() +=
                        as_vector_r(ci).array() * as_vector_r(bp).array() * i;
            }
        }

        if (!initialized) {
            Bnk.fill_with_inplace(0);
        }
        Bnk *= fp_t(k) / fp_t(n);

        return Bnk;
    }

    void invalidate(int n, int k) { m_bell_fac.at(n - 1).at(k - 1).clear(); }
};
#endif

using namespace sanm;
using namespace symbolic;

struct AnalyticUnaryOprMeta::StateCache final : public VarNodeExeCtx::Userdata {
    TensorND k;          //!< cached slope matrix, which is f'(x0)
    TensorND self_bias;  //!< bias w.r.t. input (not the graph input)
    UnaryAnalyticTrait::TaylorCoeffUserDataPtr trait_user_data;
};

void AnalyticUnaryOprMeta::eval_bias(OperatorNode* opr,
                                     ExecutionContext& ctx) const {
    const auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    sanm_assert(ictx.coeffs.size() == 1 && octx.coeffs.size() == 0);
    octx.coeffs.emplace_back(param(opr)->trait->eval(ictx.coeffs[0]));
}

void AnalyticUnaryOprMeta::accum_inp_grad(OperatorNode* opr,
                                          ExecutionContext& ctx) const {
    auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));

    StateCache& state_cache = octx.create_user_data<StateCache>();
    auto trait = param(opr)->trait.get();
    trait->eval_derivative(state_cache.k, ictx.coeffs[0]);
    ictx.jacobian += octx.jacobian.compose_with_elemwise(state_cache.k);
}

void AnalyticUnaryOprMeta::compute_order_bias(OperatorNode* opr,
                                              ExecutionContext& ctx) const {
    auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    StateCache& state_cache = octx.get_user_data<StateCache>();
    auto trait = param(opr)->trait.get();
    state_cache.self_bias.clear();
    trait->prop_taylor_coeff(state_cache.self_bias, octx.coeffs, ictx.coeffs,
                             &state_cache.trait_user_data);
    octx.cur_order_bias.as_fma(state_cache.k, ictx.cur_order_bias,
                               state_cache.self_bias);
}

void AnalyticUnaryOprMeta::compute_coeff(OperatorNode* opr,
                                         ExecutionContext& ctx) const {
    auto& ictx = ctx.get(opr->input(0));
    auto& octx = ctx.get(opr->output(0));
    StateCache& state_cache = octx.get_user_data<StateCache>();
    octx.coeffs.emplace_back().as_fma(ictx.coeffs.back(), state_cache.k,
                                      state_cache.self_bias);
}

const AnalyticUnaryOprMeta* AnalyticUnaryOprMeta::instance() {
    static AnalyticUnaryOprMeta inst;
    return &inst;
}

VarNode* AnalyticUnaryOprMeta::make(VarNode* x, UnaryAnalyticTraitPtr trait) {
    std::unique_ptr<Param> param{new Param{std::move(trait)}};
    auto opr = x->owner_graph()->insert_opr(instance(), param.get(), {x});
    param.release();
    return opr->output(0);
}
