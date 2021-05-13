/**
 * \file libsanm/stl.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

//! STL: SANM Template Library
#pragma once

#include "libsanm/utils.h"

#include <algorithm>
#include <span>

namespace sanm {

//! a dynamically sized array of objects with custom initialization
template <typename Obj, bool ctor_append_id = false>
class ObjArray : public NonCopyable {
    Obj* m_ptr;
    size_t m_size, m_ctor_called = 0;

public:
    template <typename... Args>
    explicit ObjArray(size_t size, Args&&... args)
            : m_ptr{static_cast<Obj*>(malloc(size * sizeof(Obj)))},
              m_size{size} {
        sanm_assert(m_ptr, "malloc failed");
        for (; m_ctor_called < size; ++m_ctor_called) {
            if constexpr (ctor_append_id) {
                new (&m_ptr[m_ctor_called])
                        Obj(std::forward<Args>(args)..., m_ctor_called);
            } else {
                new (&m_ptr[m_ctor_called]) Obj(std::forward<Args>(args)...);
            }
        }
    }

    ~ObjArray() {
        if (m_ptr) {
            for (size_t i = 0; i < m_ctor_called; ++i) {
                m_ptr[i].~Obj();
            }
            free(m_ptr);
            m_ptr = nullptr;
            m_ctor_called = 0;
        }
    }

    size_t size() const { return m_size; }

    //! implicit conversion to std::span
    operator std::span<Obj>() { return {m_ptr, m_size}; }
    operator std::span<const Obj>() const { return {m_ptr, m_size}; }

    //! access by index, without boundary check
    Obj& operator[](size_t i) { return m_ptr[i]; }
    const Obj& operator[](size_t i) const { return m_ptr[i]; }

    Obj* begin() { return m_ptr; }
    const Obj* begin() const { return m_ptr; }
    Obj* end() { return m_ptr + m_size; }
    const Obj* end() const { return m_ptr + m_size; }
};

namespace details {
template <typename T>
struct ReverseAdaptor {
    T* iterable;

    auto begin() const { return iterable->rbegin(); }

    auto end() const { return iterable->rend(); }
};
}  // namespace details

template <typename T>
details::ReverseAdaptor<T> reverse(T& iterable) {
    return {&iterable};
}

}  // namespace sanm
