/**
 * \file libsanm/symbolic.cpp
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#include "libsanm/symbolic.h"
#include "libsanm/oprs/misc.h"
#include "libsanm/stl.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>

using namespace sanm;
using namespace symbolic;

/* ======================= global functions ======================= */
OprNodeArray symbolic::topo_sort(const VarNodeArray& dest) {
    sanm_assert(!dest.empty());
    OprNodeArray ret;
    std::unordered_set<OperatorNode*> visited;
    std::function<void(VarNode*)> dfs;
    dfs = [&ret, &visited, &dfs](VarNode* var) {
        sanm_assert(var);
        auto opr = var->owner_opr();
        if (!visited.insert(opr).second) {
            return;
        }
        for (auto i : opr->inputs()) {
            dfs(i);
        }
        ret.push_back(opr);
    };
    for (auto i : dest) {
        dfs(i);
    }
    return ret;
}

TensorND symbolic::eval_unary_func(VarNode* y, const TensorND& xval) {
    bool input_found = false;
    ExecutionContext ectx;
    for (auto opr : topo_sort({y})) {
        if (opr->isinstance<PlaceholderOprMeta>()) {
            sanm_assert(!input_found, "multiple input variables");
            input_found = true;
            ectx.m_var2ctx[opr->output(0)].coeffs.emplace_back(xval);
        }
        for (VarNode& i : opr->outputs()) {
            ectx.m_var2ctx[&i];  // create output ctx
        }
        opr->meta()->infer_shape_eval_bias(opr, ectx);
    }
    sanm_assert(input_found, "input not found");
    return ectx.get(y).coeffs.at(0);
}

/* ======================= OperatorNode ======================= */
OperatorNode::OperatorNode(ComputingGraph* cg, const OperatorMeta* meta,
                           void* storage, VarNodeArray inputs)
        : m_cg{cg},
          m_meta{meta},
          m_id{cg->next_obj_id()},
          m_storage{storage},
          m_inputs{std::move(inputs)} {
    try {
        sanm_assert(m_inputs.size() == meta->nr_input(this),
                    "opr %s: expected %zu inputs, got %zu", meta->name(),
                    meta->nr_input(this), m_inputs.size());
        for (auto i : m_inputs) {
            sanm_assert(i->owner_graph() == m_cg,
                        "mixing oprs from different graphs");
        }
        auto nr_out = meta->nr_output(this);
        sanm_assert(nr_out, "opr must have at least one output");
        m_nr_outputs = nr_out;
        m_outputs_storage.reset(new VarNode[nr_out]);
        for (size_t i = 0; i < nr_out; ++i) {
            m_outputs_storage[i].init(this);
        }
    } catch (...) {
        m_storage = nullptr;
        throw;
    }
}

void OperatorNode::on_visit_invalid_input(size_t i) const {
    throw SANMError{ssprintf("accessing input var %zu/%zu on %s", i,
                             m_inputs.size(), str().c_str())};
}

void OperatorNode::on_visit_invalid_output(size_t i) const {
    throw SANMError{ssprintf("accessing output var %zu/%zu on %s", i,
                             m_nr_outputs, str().c_str())};
}

std::string OperatorNode::str() const {
    return ssprintf("Opr{%s@id=%zu}", m_meta->name(), m_id);
}

OperatorNode::~OperatorNode() {
    if (m_storage) {
        m_meta->on_opr_del(this);
        m_storage = nullptr;
    }
}

/* ======================= ComputingGraph ======================= */
OperatorNode* ComputingGraph::insert_opr(const OperatorMeta* meta,
                                         void* storage, VarNodeArray inputs) {
    m_oprs.emplace_back(
            new OperatorNode{this, meta, storage, std::move(inputs)});
    return m_oprs.back().get();
}

/* ======================= TensorValueMap ======================= */
TensorValueMap& TensorValueMap::insert(VarNode* var, TensorND value) {
    auto opr = var->owner_opr();
    sanm_assert(opr->isinstance<PlaceholderOprMeta>(),
                "TensorValueMap key has invalid opr type: %s",
                opr->str().c_str());
    auto ins = m_map.emplace(var, std::move(value));
    sanm_assert(ins.second, "duplicated var: %s", opr->str().c_str());
    return *this;
}

TensorArray TensorValueMap::pack(std::span<VarNode* const> vars) const {
    TensorArray ret;
    ret.reserve(vars.size());
    for (VarNode* i : vars) {
        ret.emplace_back(get(i));
    }
    return ret;
}

/* ======================= TaylorCoeffProp ======================= */

TaylorCoeffProp::TaylorCoeffProp(VarNode* output, bool output_is_batched)
        : m_topo_order{topo_sort({output})},
          m_output_is_batched{output_is_batched},
          m_output_var{output} {
    m_exe_ctx.m_var2ctx.reserve(m_topo_order.size());
    for (auto opr : m_topo_order) {
        for (VarNode& i : opr->outputs()) {
            // create the storage for VarNodeEctx
            m_exe_ctx.m_var2ctx[&i];
        }
        for (VarNode* i : opr->inputs()) {
            ++m_exe_ctx.get(i).m_nr_reader;
        }
        if (opr->isinstance<PlaceholderOprMeta>()) {
            m_input_vars.emplace_back(opr->output(0));
        }
    }
    sanm_assert(!m_input_vars.empty(), "no input var not found");
}

const TensorND& TaylorCoeffProp::push_xi(std::span<const TensorND> inp_val) {
    sanm_assert(!m_xi_known);
    sanm_assert(inp_val.size() == m_input_vars.size(),
                "expect %zu inputs, got %zu", m_input_vars.size(),
                inp_val.size());
    for (size_t i = 0; i < m_input_vars.size(); ++i) {
        auto& ictx = m_exe_ctx.get(m_input_vars[i]);
        if (m_exe_ctx.order()) {
            sanm_assert(ictx.shape == inp_val[i].shape());
        }
        ictx.coeffs.emplace_back(inp_val[i]);
    }

    for (OperatorNode* opr : m_topo_order) {
        if (m_exe_ctx.order() == 0) {
            // compute the first coeff value, which is f(x0)
            opr->meta()->infer_shape_eval_bias(opr, m_exe_ctx);
        } else {
            opr->meta()->compute_coeff(opr, m_exe_ctx);
        }

        // sanity check of the outputs
        for (VarNode& i : opr->outputs()) {
            auto& vctx = m_exe_ctx.get(i);
            if (!vctx.nr_reader()) {
                continue;
            }
            sanm_assert(vctx.shape.rank);
            sanm_assert(vctx.coeffs.size() == m_exe_ctx.order() + 1,
                        "opr %s: order=%zu coeffs_size=%zu", opr->str().c_str(),
                        m_exe_ctx.order(), vctx.coeffs.size());
            sanm_assert(vctx.coeffs.back().shape() == vctx.shape,
                        "opr %s: declared output var shape(%s) mismatches "
                        "with the shape of computed coefficient (%s) for "
                        "order %zu ",
                        opr->str().c_str(), vctx.shape.str().c_str(),
                        vctx.coeffs.back().shape().str().c_str(),
                        m_exe_ctx.order());
        }
    }
    m_xi_known = true;
    return m_exe_ctx.get(m_output_var).coeffs.back();
}

void TaylorCoeffProp::ensure_jacobian() {
    if (m_jacobian_done) {
        return;
    }
    sanm_assert(m_exe_ctx.order() == 0);
    auto& out_ctx = m_exe_ctx.get(m_output_var);
    TensorShape out_shape;
    if (m_output_is_batched) {
        out_shape = out_ctx.shape.flatten_batched();
    } else {
        out_shape.rank = 1;
        out_shape.dim[0] = out_ctx.shape.total_nr_elems();
    }
    out_ctx.jacobian.reset(StSparseLinearTrans::ELEMWISE, m_output_is_batched,
                           TensorND{out_shape}.fill_with_inplace(1));

    for (OperatorNode* opr : reverse(m_topo_order)) {
        opr->meta()->accum_inp_grad(opr, m_exe_ctx);
        for (VarNode* i : opr->inputs()) {
            auto& ictx = m_exe_ctx.get(i);
            sanm_assert(ictx.jacobian.valid());
            if (ictx.jacobian.is_batched()) {
                sanm_assert(ictx.jacobian.batch() == ictx.shape[0] &&
                            ictx.jacobian.inp_dim() ==
                                    ictx.shape.total_nr_elems_per_batch());
            } else {
                sanm_assert(ictx.jacobian.inp_dim() ==
                            ictx.shape.total_nr_elems());
            }
        }
    }

    // reclaim memory of Jacobian values unused in the future
    for (OperatorNode* opr : m_topo_order) {
        if (!opr->isinstance<PlaceholderOprMeta>()) {
            for (VarNode& v : opr->outputs()) {
                m_exe_ctx.get(v).jacobian = {};
            }
        }
    }
    m_jacobian_done = true;
}

const TensorND& TaylorCoeffProp::compute_next_order_bias() {
    SANM_SCOPED_PROFILER("taylor_next_order");
    ensure_jacobian();

    sanm_assert(m_xi_known);
    ++m_exe_ctx.m_order;
    m_xi_known = false;

    for (auto& i : m_exe_ctx.m_var2ctx) {
        sanm_assert(!i.second.nr_reader() ||
                    i.second.coeffs.size() ==
                            static_cast<size_t>(m_exe_ctx.order()));
        i.second.cur_order_bias.clear();
    }

    for (OperatorNode* opr : m_topo_order) {
        opr->meta()->compute_order_bias(opr, m_exe_ctx);

        // sanity check of the outputs
        for (VarNode& i : opr->outputs()) {
            auto& vctx = m_exe_ctx.get(&i);
            if (!vctx.nr_reader()) {
                continue;
            }
            sanm_assert(vctx.cur_order_bias.shape() == vctx.shape,
                        "opr %s: bias shape mismatch: %s vs %s",
                        opr->str().c_str(),
                        vctx.cur_order_bias.shape().str().c_str(),
                        vctx.shape.str().c_str());
            if (m_exe_ctx.m_order == 1) {
                // Bias of the first term is always zero. Here we still call
                // compute_order_bias() to initialize internal data structures
                // for next call of compute_coeff()
                sanm_assert(vctx.cur_order_bias.is_zero(),
                            "opr %s: bias is non-zero for first order",
                            opr->str().c_str());
            }
        }
    }
    return get_prev_next_order_bias();
}

const TensorND& TaylorCoeffProp::get_prev_next_order_bias() const {
    sanm_assert(!m_xi_known);
    return const_cast<TaylorCoeffProp*>(this)
            ->m_exe_ctx.get(m_output_var)
            .cur_order_bias;
}

const StSparseLinearTrans& TaylorCoeffProp::get_jacobian(VarNode* x) {
    ensure_jacobian();
    const auto& ctx = const_cast<ExecutionContext&>(m_exe_ctx).get(x);
    sanm_assert(ctx.jacobian.valid());
    return ctx.jacobian;
}

/* ======================= ParallelTaylorCoeffProp ======================= */
class ParallelTaylorCoeffProp::Impl {
    class Worker {
        TaylorCoeffProp m_solver;
        std::thread m_thread;
        TensorND m_xi;
        const TensorND *m_yi = nullptr, *m_next_order_bias = nullptr;

        void mainloop(Impl* owner, size_t self_id);

    public:
        Worker(VarNode* var, Impl* owner, size_t self_id)
                : m_solver{var, true},
                  m_thread{[this, owner, self_id,
                            prof_node = ScopedProfiler::get_node()]() {
                      try {
                          ScopedProfiler::set_thread_root_node(prof_node);
                          SANM_DEFER(ScopedProfiler::clear_thread_root_node);
                          mainloop(owner, self_id);
                      } catch (std::exception& exc) {
                          fprintf(stderr,
                                  "caught exception in "
                                  "ParallelTaylorCoeffProp worker %zu: "
                                  "what(): %s\n",
                                  self_id, exc.what());
                          std::lock_guard<std::mutex> lock{owner->m_mtx};
                          owner->m_worker_except = std::current_exception();
                          owner->m_cvar_finish.notify_one();
                      } catch (...) {
                          std::lock_guard<std::mutex> lock{owner->m_mtx};
                          owner->m_worker_except = std::current_exception();
                          owner->m_cvar_finish.notify_one();
                      }
                  }} {}

        void join() { m_thread.join(); }

        const TensorND& yi() const {
            sanm_assert(m_yi);
            return *m_yi;
        }

        const TensorND& next_order_bias() const {
            sanm_assert(m_next_order_bias);
            return *m_next_order_bias;
        }

        const StSparseLinearTrans& get_jacobian() {
            auto&& inp = m_solver.inputs();
            sanm_assert(inp.size() == 1);
            return m_solver.get_jacobian(inp[0]);
        }
    };

    enum class JobType { NONE, PUSH, NEXT_ORDER_BIAS, CALL, QUIT };

    const TensorND* m_xi_inp = nullptr;  //!< current push_xi() argument
    std::mutex m_mtx;
    std::condition_variable m_cvar_new_job, m_cvar_finish;
    JobType m_job_type = JobType::NONE;
    size_t m_job_id = 0, m_nr_finished = 0, m_batch_size = 0;
    TensorND m_next_order_bias;
    const WorkerTaskFn* m_worker_callback = nullptr;
    std::exception_ptr m_worker_except;

    // must be the last member: last ctor, first dtor
    ObjArray<Worker, true> m_workers;

    void check_exception() {
        if (m_worker_except) {
            std::rethrow_exception(m_worker_except);
        }
    }

    void submit_job(JobType job) {
        std::lock_guard<std::mutex> lock{m_mtx};
        check_exception();
        sanm_assert(!m_nr_finished);
        m_job_type = job;
        ++m_job_id;
        m_cvar_new_job.notify_all();
    }

    void wait_finish() {
        std::unique_lock<std::mutex> lock{m_mtx};
        for (;;) {
            check_exception();
            if (m_nr_finished == m_workers.size()) {
                m_job_type = JobType::NONE;
                m_nr_finished = 0;
                return;
            }
            m_cvar_finish.wait(lock);
        }
    }

    //! id starts from 1
    JobType worker_fetch_job(size_t id) {
        std::unique_lock<std::mutex> lock{m_mtx};
        for (;;) {
            if (m_job_id == id) {
                return m_job_type;
            }
            m_cvar_new_job.wait(lock);
        }
    }

    void worker_mark_finish() {
        std::lock_guard<std::mutex> lock{m_mtx};
        ++m_nr_finished;
        m_cvar_finish.notify_one();
    }

    void gather_into(TensorND& dst, size_t i, const TensorND& src) const {
        if (m_workers.size() == 1) {
            dst = src;
            return;
        }

        if (!i) {
            dst.set_shape(src.shape().with_batch_replaced(m_batch_size));
            if (src.is_zero()) {
                // assume all parts are zero, and check later
                dst.fill_with_inplace(0);
                return;
            }
        }
        if (src.is_zero() && dst.is_zero()) {
            return;
        }
        size_t begin = i * m_batch_size / m_workers.size(),
               end = (i + 1) * m_batch_size / m_workers.size();
        sanm_assert(src.shape(0) == end - begin);
        dst.copy_from_sub_batch(src, begin, 0, end - begin);
    }

public:
    void push_xi(const TensorND& val) {
        m_next_order_bias.clear();
        m_xi_inp = &val;
        submit_job(JobType::PUSH);
        wait_finish();
        m_xi_inp = nullptr;
        m_batch_size = val.shape(0);
    }

    TensorND gather_yi() const {
        TensorND ret;
        for (size_t i = 0; i < m_workers.size(); ++i) {
            gather_into(ret, i, m_workers[i].yi());
        }
        return ret;
    }

    const TensorND& compute_next_order_bias() {
        submit_job(JobType::NEXT_ORDER_BIAS);
        wait_finish();
        if (m_workers.size() == 1) {
            return m_workers[0].next_order_bias();
        }
        for (size_t i = 0; i < m_workers.size(); ++i) {
            gather_into(m_next_order_bias, i, m_workers[i].next_order_bias());
        }
        return m_next_order_bias;
    }

    const TensorND& get_prev_next_order_bias() {
        if (m_workers.size() == 1) {
            return m_workers[0].next_order_bias();
        }
        sanm_assert(!m_next_order_bias.empty());
        return m_next_order_bias;
    }

    StSparseLinearTrans get_jacobian() {
        TensorND coeff;
        StSparseLinearTrans::Type type = StSparseLinearTrans::INVALID;
        for (size_t i = 0; i < m_workers.size(); ++i) {
            const auto& lti = m_workers[i].get_jacobian();
            if (!i) {
                type = lti.type();
            } else {
                sanm_assert(type == lti.type());
            }
            gather_into(coeff, i, lti.coeff());
        }
        return {type, true, coeff};
    }

    size_t nr_worker() const { return m_workers.size(); }

    void run_on_workers(const WorkerTaskFn& fn) {
        if (m_workers.size() == 1) {
            fn(0);
            return;
        }

        m_worker_callback = &fn;
        submit_job(JobType::CALL);
        wait_finish();
        m_worker_callback = nullptr;
    }

    explicit Impl(VarNode* var)
            : m_workers{static_cast<size_t>(get_num_threads()), var, this} {}

    ~Impl() {
        submit_job(JobType::QUIT);
        for (auto& i : m_workers) {
            i.join();
        }
    }
};

void ParallelTaylorCoeffProp::Impl::Worker::mainloop(Impl* owner,
                                                     size_t self_id) {
    for (size_t job_id = 1;; ++job_id) {
        switch (owner->worker_fetch_job(job_id)) {
            case JobType::PUSH: {
                m_next_order_bias = nullptr;
                size_t size = owner->m_xi_inp->shape(0),
                       nr = owner->m_workers.size(),
                       begin = self_id * size / nr,
                       end = (self_id + 1) * size / nr;
                m_solver.m_exe_ctx.m_parallel_shard.init(begin, end, size);
                if (nr == 1) {
                    m_yi = &m_solver.push_xi(*owner->m_xi_inp);
                } else {
                    m_xi.copy_from_sub_batch(*owner->m_xi_inp, 0, begin,
                                             end - begin);
                    m_yi = &m_solver.push_xi(m_xi);
                }
                break;
            }
            case JobType::NEXT_ORDER_BIAS: {
                m_yi = nullptr;
                m_next_order_bias = &m_solver.compute_next_order_bias();
                break;
            }
            case JobType::CALL: {
                (*owner->m_worker_callback)(self_id);
                break;
            }
            case JobType::QUIT: {
                return;
            }
            default:
                sanm_assert(0);
        }
        owner->worker_mark_finish();
    }
}

ParallelTaylorCoeffProp::ParallelTaylorCoeffProp(VarNode* var)
        : m_pimpl{new Impl{var}} {}

ParallelTaylorCoeffProp::~ParallelTaylorCoeffProp() = default;

ParallelTaylorCoeffProp& ParallelTaylorCoeffProp::push_xi(const TensorND& val) {
    m_pimpl->push_xi(val);
    return *this;
}

const TensorND& ParallelTaylorCoeffProp::compute_next_order_bias() {
    return m_pimpl->compute_next_order_bias();
}

const TensorND& ParallelTaylorCoeffProp::get_prev_next_order_bias() const {
    return m_pimpl->get_prev_next_order_bias();
}

StSparseLinearTrans ParallelTaylorCoeffProp::get_jacobian() {
    return m_pimpl->get_jacobian();
}

TensorND ParallelTaylorCoeffProp::gather_yi() const {
    return m_pimpl->gather_yi();
}

size_t ParallelTaylorCoeffProp::nr_worker() const {
    return m_pimpl->nr_worker();
}

void ParallelTaylorCoeffProp::run_on_workers(const WorkerTaskFn& fn) {
    m_pimpl->run_on_workers(fn);
}
