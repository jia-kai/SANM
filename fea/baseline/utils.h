#pragma once

#include <chrono>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include <cstdarg>

#ifndef CF_USE_TIMER
#define CF_USE_TIMER 0
#endif

namespace cf {

#define cf_assert(_expr, ...)                                              \
    do {                                                                   \
        if (!(_expr)) {                                                    \
            ::cf::__assertion_failed(#_expr, __FILE__, __func__, __LINE__, \
                                     ##__VA_ARGS__);                       \
        }                                                                  \
    } while (0)

#define CF_PASTE2(a, b) a##b
#define CF_PASTE(a, b) CF_PASTE2(a, b)

//! declare a var to be executed at the exit of the scope
#define CF_DEFER(func) \
    ::cf::DeferExec CF_PASTE(__defer, __LINE__) { func }

class CFError : public std::exception {
    std::string m_msg;

public:
    explicit CFError(std::string msg) : m_msg{std::move(msg)} {}
    const char* what() const noexcept override { return m_msg.c_str(); }
};

class CFAssertionError : public CFError {
public:
    using CFError::CFError;
};

class NonCopyable {
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;

protected:
    NonCopyable() = default;
};

class DeferExec : public NonCopyable {
    std::function<void()> m_func;

public:
    explicit DeferExec(std::function<void()> func) : m_func{std::move(func)} {}
    ~DeferExec() { m_func(); }
};

[[noreturn]] void __assertion_failed(const char* expr, const char* file,
                                     const char* func, int line,
                                     const char* fmt = nullptr, ...)
        __attribute__((format(printf, 5, 6)));

std::string ssprintf(const char* fmt, ...)
        __attribute__((format(printf, 1, 2)));
std::string svsprintf(const char* fmt, va_list ap);

class NoneType {};
//! represent absence of value in Maybe
static constexpr NoneType None;

//! an optional value
template <typename T>
class Maybe {
    bool m_valid = false;
    typename std::aligned_storage<sizeof(T), alignof(T)>::type m_storage;

    T* ptr() { return reinterpret_cast<T*>(&m_storage); }
    const T* ptr() const { return reinterpret_cast<const T*>(&m_storage); }

public:
    Maybe() = default;

    Maybe(NoneType) {}

    // inplace constructor
    template <typename... C,
              typename = std::enable_if<std::is_constructible<T, C...>::value,
                                        void>>
    Maybe(C&&... c) {
        m_valid = true;
        new (&m_storage) T(std::forward<C>(c)...);
    }

    Maybe(const Maybe& rhs) { *this = rhs; }
    Maybe(Maybe&& rhs) { *this = std::move(rhs); }

    ~Maybe() { reset(); }

    //! clear the storage and release the object
    void reset() {
        if (m_valid) {
            m_valid = false;
            ptr()->~T();
        }
    }

    //! initialize as a new value
    template <typename... C>
    std::enable_if<std::is_constructible<T, C...>::value, void> init(C&&... c) {
        reset();
        m_valid = true;
        new (&m_storage) T(std::forward<C>(c)...);
    }

    Maybe& operator=(const Maybe& rhs) noexcept(
            std::is_nothrow_copy_assignable<T>::value) {
        reset();
        m_valid = rhs.m_valid;
        if (m_valid) {
            new (&m_storage) T(*rhs.ptr());
        }
        return *this;
    }

    Maybe& operator=(Maybe&& rhs) noexcept(
            std::is_nothrow_move_assignable<T>::value) {
        if (m_valid && rhs.m_valid) {
            (*ptr()) = std::move(*rhs.ptr());
            return *this;
        }
        reset();
        m_valid = rhs.m_valid;
        if (m_valid) {
            new (&m_storage) T(std::move(*rhs.ptr()));
        }
        return *this;
    }

    bool valid() const { return m_valid; }

    T& val() {
        cf_assert(m_valid);
        return *ptr();
    }
    const T& val() const { return const_cast<Maybe*>(this)->val(); }

    T* operator->() { return &val(); }
    const T* operator->() const { return &val(); }
};

template <typename T>
static inline size_t hash_combine(size_t seed, const T& v) {
    // Code from boost
    // Reciprocal of the golden ratio helps spread entropy
    //     and handles duplicates.
    // See Mike Seymour in magic-numbers-in-boosthash-combine:
    //     http://stackoverflow.com/questions/4948780
    seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

class HashChain {
    size_t m_seed;

public:
    explicit HashChain(size_t seed = 0) : m_seed{seed} {}

    template <typename T>
    HashChain& feed(T&& v) {
        m_seed = hash_combine(m_seed, std::forward<T>(v));
        return *this;
    }

    size_t v() const { return m_seed; }
};

#if CF_USE_TIMER
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    bool m_started = false;
    decltype(Clock::now()) m_start_time;
    double m_acc_time = 0;

public:
    void start() {
        cf_assert(!m_started);
        m_started = true;
        m_start_time = Clock::now();
    }
    void stop() {
        cf_assert(m_started);
        m_started = false;
        m_acc_time += static_cast<std::chrono::duration<double>>(Clock::now() -
                                                                 m_start_time)
                              .count();
    }

    //! accumulated time in seconds
    double time() const { return m_acc_time; }

    void reset() { m_acc_time = 0; }
};
#else
class Timer {
public:
    void start() {}
    void stop() {}
    double time() const { return 0; }
    void reset() {}
};
#endif

// UniformRandomBitGenerator implementation using xorshift128+
class Xorshift128pRng {
    uint64_t m_a, m_b;

public:
    Xorshift128pRng(uint64_t seed = 0);

    using result_type = uint64_t;

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() {
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t operator()() {
        uint64_t t = m_a;
        uint64_t const s = m_b;
        m_a = s;
        t ^= t << 23;        // a
        t ^= t >> 17;        // b
        t ^= s ^ (s >> 26);  // c
        m_b = t;
        return t + s;
    }
};

}  // namespace cf

