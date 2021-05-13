/**
 * \file libsanm/analytic_unary.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/analytic_unary.h"

#include <cmath>

using namespace sanm;

namespace {
class LogImpl final : public UnaryAnalyticTrait {
public:
    void eval(TensorND& dst, const TensorND& src) const override {
        setup_dst(dst, src);
        dst.as_log(src);
    }

    void eval_derivative(TensorND& dst, const TensorND& src) const override {
        setup_dst(dst, src);
        dst.as_pow(src, -1);
    }

    void do_prop_taylor_coeff(TensorND& dst, const TensorArray& f,
                              const TensorArray& x,
                              TaylorCoeffUserDataPtr*) const override {
        size_t k = f.size();
        for (size_t i = 1; i < k; ++i) {
            dst.accum_mul(x[k - i], f[i], -fp_t(i) / fp_t(k));
        }
        dst /= x[0];  // no need to worry about zero because can not log(0)
    }
};

class PowImpl final : public UnaryAnalyticTrait {
    const fp_t m_exp;

    struct UserData final : public TaylorCoeffUserData {
        bool has_zero = false;
    };

    static bool is_zero(fp_t x) { return std::fabs(x) < 1e-3; }

    //! propagate for integer exponents when x is zero
    static void prop_taylor_coeff_int(TensorND& dst, const TensorArray& x,
                                      int exp) {
        auto conv = [k = x.size()](TensorArray& dst, const TensorArray& x,
                                   const TensorArray& y) {
            dst.resize(k + 1);
            for (auto&& i : dst) {
                i.set_shape(x[0].shape()).fill_with_inplace(0);
            }
            for (size_t i = 0; i < x.size(); ++i) {
                for (size_t j = 0; j < y.size() && (i + j) <= k; ++j) {
                    dst[i + j].accum_mul(x[i], y[j]);
                }
            }
        };
        auto conv_k = [k = x.size(), &dst](const TensorArray& x,
                                           const TensorArray& y) {
            dst.set_shape(x[0].shape()).fill_with_inplace(0);
            for (size_t i = std::max<int>(0, int(k) + 1 - int(y.size()));
                 i < x.size() && i <= k; ++i) {
                dst.accum_mul(x[i], y[k - i]);
            }
        };

        TensorArray buf[4], *xi = &buf[0], *xi_next = &buf[1], *prod = &buf[2],
                            *prod_next = &buf[3];
        *xi = x;

        while (exp > 1) {
            if (exp % 2) {
                if (prod->empty()) {
                    *prod = *xi;
                } else {
                    conv(*prod_next, *prod, *xi);
                    std::swap(prod, prod_next);
                }
            }
            if (exp == 2 && prod->empty()) {
                conv_k(*xi, *xi);
                return;
            }
            exp /= 2;
            conv(*xi_next, *xi, *xi);
            std::swap(xi, xi_next);
        }
        sanm_assert(!prod->empty());
        conv_k(*prod, *xi);
    }

public:
    explicit PowImpl(fp_t exp) : m_exp{exp} {}

    void eval(TensorND& dst, const TensorND& src) const override {
        dst.as_pow(src, m_exp);
    }

    void eval_derivative(TensorND& dst, const TensorND& src) const override {
        dst = src.pow(m_exp - 1) * m_exp;
    }

    void do_prop_taylor_coeff(
            TensorND& dst, const TensorArray& f, const TensorArray& x,
            TaylorCoeffUserDataPtr* user_data_p) const override {
        size_t k = f.size();

        auto x0ptr = x[0].ptr();
        sanm_assert(user_data_p, "computing without user data unsupported");
        if (!*user_data_p) {
            auto ud = std::make_unique<UserData>();
            for (size_t i = 0, it = x[0].shape().total_nr_elems(); i < it;
                 ++i) {
                if (is_zero(x0ptr[i])) {
                    if (m_exp <= 0.5 || std::floor(m_exp) != m_exp) {
                        throw SANMNumericalError{ssprintf(
                                "0^p when p is not integer: %g", m_exp)};
                    }
                    ud->has_zero = true;
                    break;
                }
            }
            *user_data_p = std::move(ud);
        }
        const UserData& ud = dynamic_cast<UserData&>(**user_data_p);
        if (ud.has_zero) {
            prop_taylor_coeff_int(dst, x, m_exp);
            return;
        }

        for (size_t i = 1; i < k; ++i) {
            dst.accum_mul(f[k - i], x[i], fp_t(i) / fp_t(k) * (m_exp + 1) - 1);
        }

        dst /= x[0];
    }
};
}  // anonymous namespace

fp_t* UnaryAnalyticTrait::setup_dst(TensorND& dst, const TensorND& src) {
    if (&dst == &src) {
        return dst.rwptr();
    }
    return dst.set_shape(src.shape()).woptr();
}

void UnaryAnalyticTrait::prop_taylor_coeff(
        TensorND& dst, const TensorArray& f, const TensorArray& x,
        TaylorCoeffUserDataPtr* user_data_p) const {
    sanm_assert(!f.empty() && f.size() == x.size() &&
                f[0].shape() == x[0].shape());
    dst.clear();
    if (f.size() == 1) {
        dst.set_shape({f[0].shape()}).fill_with_inplace(0);
    } else {
        do_prop_taylor_coeff(dst, f, x, user_data_p);
    }
}

UnaryAnalyticTraitPtr UnaryAnalyticTrait::make_log() {
    static auto ret = std::make_shared<LogImpl>();
    return ret;
}

UnaryAnalyticTraitPtr UnaryAnalyticTrait::make_pow(fp_t exp) {
    sanm_assert(std::fabs(exp) > 1e-9, "zero power not handled");
    return std::make_shared<PowImpl>(exp);
}
