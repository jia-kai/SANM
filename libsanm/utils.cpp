/**
 * \file libsanm/utils.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/utils.h"

#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using namespace sanm;

std::string sanm::ssprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    SANM_DEFER([&ap]() { va_end(ap); });
    auto ret = svsprintf(fmt, ap);
    return ret;
}

std::string sanm::svsprintf(const char* fmt, va_list ap_orig) {
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

void sanm::__assertion_failed(const char* expr, const char* file,
                              const char* func, int line, const char* fmt,
                              ...) {
    auto msg = ssprintf("assertion `%s' failed at %s:%s@%d", expr, file, func,
                        line);
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        SANM_DEFER([&ap]() { va_end(ap); });
        msg += "\nmessage: ";
        msg += svsprintf(fmt, ap);
    }
    throw SANMAssertionError{std::move(msg)};
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

/* ======================= ScopedProfiler ======================= */

struct ScopedProfiler::Stat {
    size_t nr_call = 0;
    double min_time = std::numeric_limits<double>::max(), max_time = 0,
           tot_time = 0;

    void update(double t) {
        ++nr_call;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
        tot_time += t;
    }
    std::unordered_map<TagId, std::unique_ptr<Stat>> sub;

    Stat* get_sub(TagId tid) {
        auto ins = sub.emplace(tid, nullptr);
        if (ins.second) {
            ins.first->second.reset(new Stat);
        }
        return ins.first->second.get();
    }
};

class ScopedProfiler::Recorder final : public NonCopyable {
public:
    void enter(const char* tag) {
        std::lock_guard<std::mutex> lg{m_mutex};
        TagId tid;
        {
            auto ins = m_tag2id.emplace(tag, m_tag2id.size());
            if (ins.second) {
                m_tags.emplace_back(tag);
            }
            tid = ins.first->second;
        }
        auto& call_stack = m_call_stack[std::this_thread::get_id()];
        Stat* root;
        if (call_stack.empty()) {
            root = &m_root;
        } else {
            root = call_stack.back();
        }
        Stat* stat = root->get_sub(tid);
        call_stack.emplace_back(stat);
    }

    void exit(double time) {
        std::lock_guard<std::mutex> lg{m_mutex};
        auto call_stack_iter = m_call_stack.find(std::this_thread::get_id());
        sanm_assert(call_stack_iter != m_call_stack.end());
        auto& call_stack = call_stack_iter->second;
        call_stack.back()->update(time);
        call_stack.pop_back();
        if (call_stack.empty()) {
            m_call_stack.erase(call_stack_iter);
        }
    }

    Stat* get_node() {
        std::lock_guard<std::mutex> lock{m_mutex};
        auto iter = m_call_stack.find(std::this_thread::get_id());
        if (iter != m_call_stack.end() && !iter->second.empty()) {
            return iter->second.back();
        }
        return nullptr;
    }

    void set_thread_root_node(Stat* node) {
        if (!node) {
            return;
        }
        std::lock_guard<std::mutex> lock{m_mutex};
        auto& stack = m_call_stack[std::this_thread::get_id()];
        sanm_assert(stack.empty());
        stack.emplace_back(node);
    }

    void clear_thread_root_node() {
        std::lock_guard<std::mutex> lock{m_mutex};
        auto iter = m_call_stack.find(std::this_thread::get_id());
        sanm_assert(iter != m_call_stack.end() && iter->second.size() == 1);
        m_call_stack.erase(iter);
    }

    void report() const;

    ~Recorder() { report(); }

private:
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, TagId> m_tag2id;
    std::unordered_map<std::thread::id, std::vector<Stat*>> m_call_stack;
    std::vector<std::string> m_tags;
    Stat m_root;
};
ScopedProfiler::Recorder ScopedProfiler::sm_recorder;

ScopedProfiler::ScopedProfiler(const char* tag) {
    m_timer.start();
    sm_recorder.enter(tag);
}

ScopedProfiler::~ScopedProfiler() {
    m_timer.stop();
    sm_recorder.exit(m_timer.time());
}

void ScopedProfiler::Recorder::report() const {
    std::lock_guard<std::mutex> lg{m_mutex};
    if (m_root.sub.empty()) {
        return;
    }

    std::function<void(std::string, const Stat&)> report_one;
    report_one = [this, &report_one](std::string indent, const Stat& root) {
        using Spair = std::pair<TagId, const Stat*>;
        std::vector<Spair> stat;
        stat.reserve(root.sub.size());
        for (auto& i : root.sub) {
            stat.emplace_back(i.first, i.second.get());
        }
        std::sort(stat.begin(), stat.end(), [](const Spair& a, const Spair& b) {
            return a.second->tot_time > b.second->tot_time;
        });
        for (auto& [tid, s] : stat) {
            printf("%s%-20s: tot=%.3f avg=%.3f nr=%zu min=%.3f max=%.3f",
                   indent.c_str(), m_tags.at(tid).c_str(), s->tot_time,
                   s->tot_time / s->nr_call, s->nr_call, s->min_time,
                   s->max_time);
            if (!s->sub.empty()) {
                double sub_tot = 0;
                for (auto& [_, i] : s->sub) {
                    sub_tot += i->tot_time;
                }
                printf(" sub=%.3f/%.3f=%.0f%%", sub_tot, s->tot_time,
                       sub_tot / s->tot_time * 100);
            }
            printf("\n");
            report_one(indent + "  ", *s);
        }
    };

    printf("=========== begin profiling results ===========\n");
    report_one({}, m_root);
    printf("=========== end profiling results ===========\n");
}

ScopedProfiler::Stat* ScopedProfiler::get_node() {
    return sm_recorder.get_node();
}

void ScopedProfiler::set_thread_root_node(Stat* node) {
    sm_recorder.set_thread_root_node(node);
}

void ScopedProfiler::clear_thread_root_node() {
    sm_recorder.clear_thread_root_node();
}

void ScopedProfiler::report() {
    sm_recorder.report();
}
