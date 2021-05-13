#include "utils.h"

#include <random>
#include <stdexcept>

using namespace cf;

std::string cf::ssprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    CF_DEFER([&ap]() { va_end(ap); });
    auto ret = svsprintf(fmt, ap);
    return ret;
}

std::string cf::svsprintf(const char* fmt, va_list ap_orig) {
    int n = 0;
    va_list ap;

    /* Determine required size */
    va_copy(ap, ap_orig);
    n = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);

    if (n < 0) {
        throw std::runtime_error{"vsnprintf failed"};
    }

    /* One extra byte for '\0' */
    auto size = n + 1;
    std::string ret(size, 0);

    va_copy(ap, ap_orig);
    n = vsnprintf(&ret[0], size, fmt, ap);
    va_end(ap);

    if (n < 0) {
        throw std::runtime_error{"vsnprintf failed"};
    }

    ret.pop_back();  // remove final '\0'

    return ret;
}

void cf::__assertion_failed(const char* expr, const char* file,
                            const char* func, int line, const char* fmt, ...) {
    auto msg = ssprintf("assertion `%s' failed at %s:%s@%d", expr, file, func,
                        line);
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        CF_DEFER([&ap]() { va_end(ap); });
        msg += "\nmessage: ";
        msg += svsprintf(fmt, ap);
    }
    throw CFAssertionError{std::move(msg)};
}

Xorshift128pRng::Xorshift128pRng(uint64_t seed) {
    std::mt19937_64 rng{seed};
    for (;;) {
        m_a = rng();
        m_b = rng();
        if (m_a && m_b) {
            break;
        }
    }
}

