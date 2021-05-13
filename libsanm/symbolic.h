/**
 * \file libsanm/symbolic.h
 * This file is part of SANM, a symbolic asymptotic numerical solver.
 */

#pragma once

#include "libsanm/stl.h"
#include "libsanm/tensor.h"
#include "libsanm/utils.h"

#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sanm {
namespace symbolic {

class VarNode;
class OperatorNode;
class ComputingGraph;
class TaylorCoeffProp;

using VarNodeArray = std::vector<VarNode*>;
using OprNodeArray = std::vector<OperatorNode*>;

/*!
 * \brief Execution context for a VarNode
 *
 * This struct maintains the information associated with a VarNode during
 * graph execution.
 *
 * Extra user data can be associated with the current execution, which is
 * managed by the OperatorMeta implementation corresponding to this var.
 */
class VarNodeExeCtx {
public:
    class Userdata {
    public:
        virtual ~Userdata() = default;
    };

    //! Shape of this variable during execution, which is statically inferred.
    //! Its first dimension is the batch size.
    TensorShape shape;

    //! numerical expansion coefficients in the polynomial expansion; all
    //! coefficients have the same shape as this var
    TensorArray coeffs;

    //! bias of the expansion coefficient of current order with respect to the
    //! orignal graph inputs
    TensorND cur_order_bias;

    //! Jacobian matrix: gradient of the output w.r.t. this var
    StSparseLinearTrans jacobian;

    //! get user data
    template <typename T>
    T& get_user_data() const {
        sanm_assert(m_user_data);
        return dynamic_cast<T&>(*m_user_data);
    }

    //! create user data, which must not exist now
    template <typename T>
    T& create_user_data() {
        sanm_assert(!m_user_data);
        auto p = new T{};
        m_user_data.reset(p);
        return *p;
    }

    //! get user data, and create a new one if it does not exist
    template <typename T>
    T& get_user_data_or_create() {
        if (!m_user_data) {
            m_user_data.reset(new T{});
        }
        return dynamic_cast<T&>(*m_user_data);
    }

    //! number of readers of this var in the computing graph of TaylorCoeffProp
    size_t nr_reader() const { return m_nr_reader; }

    /*!
     * \brief get the bias tensor for computing #cur_order_bias or new coeff
     *
     * This function returns coeff.back() or #cur_order_bias according to the
     * parameter \p in_coeff. It is useful for implementing
     * OperatorMeta::compute_order_bias() and OperatorNode::compute_coeff()
     * using a shared linear transformation function.
     */
    TensorND& get_bias(bool in_coeff) {
        return in_coeff ? coeffs.back() : cur_order_bias;
    }
    const TensorND& get_bias(bool in_coeff) const {
        return const_cast<VarNodeExeCtx*>(this)->get_bias(in_coeff);
    }

private:
    friend class TaylorCoeffProp;
    size_t m_nr_reader = 0;
    std::unique_ptr<Userdata> m_user_data;
};

/*!
 * \brief representing a variable in the computing graph
 *
 * The address of a VarNode serves as its unique identifier.
 */
class VarNode : public NonCopyable {
    OperatorNode* m_owner_opr = nullptr;
    Maybe<std::string> m_name;

public:
    //! called from OperatorNode::OperatorNode() to initialize the output vars
    void init(OperatorNode* opr) {
        sanm_assert(opr && !m_owner_opr);
        m_owner_opr = opr;
    }

    OperatorNode* owner_opr() const { return m_owner_opr; }
    inline ComputingGraph* owner_graph() const;
};

//! the context of one execution
class ExecutionContext {
public:
    //! descriptor of a shard in data parallelism
    struct ShardDesc {
        size_t begin;  //!< first batch in the shard
        size_t end;    //!< one past the last batch in the shard
        //! total batch size (i.e., original batch size) before splitting
        size_t tot;

        ShardDesc(size_t b, size_t e, size_t t) : begin{b}, end{e}, tot{t} {}
    };

    //! order of term in the expansion whose coefficients are being computed
    size_t order() const { return m_order; }

    /*!
     * \brief current shard for this worker in data parallel computing
     *
     * Data provider operators (such as Constant) need to behave differently for
     * splitted input.
     */
    const Maybe<ShardDesc>& parallel_shard() const { return m_parallel_shard; }

    VarNodeExeCtx& get(VarNode* var) { return m_var2ctx.at(var); }
    VarNodeExeCtx& get(VarNode& var) { return get(&var); }

private:
    friend class TaylorCoeffProp;
    friend class ParallelTaylorCoeffProp;
    friend TensorND eval_unary_func(VarNode*, const TensorND&);
    size_t m_order = 0;
    std::unordered_map<VarNode*, VarNodeExeCtx> m_var2ctx;
    Maybe<ShardDesc> m_parallel_shard;
};

//! meta-info about a type of operator; address of this class is an identifier
//! of the operator type. Each meta class is shared by all opr instances
class OperatorMeta : public NonCopyable {
protected:
    ~OperatorMeta() = default;

public:
    //! name of this type of operator
    virtual const char* name() const = 0;

    //! needed number of input vars
    virtual size_t nr_input(OperatorNode* opr) const = 0;

    //! needed number of output vars
    virtual size_t nr_output(OperatorNode* opr) const = 0;

    /*!
     * \brief called when an operator is being destructed and its storage is
     *      non-null
     *
     * Note that this function would NOT be called if
     * ComputingGraph::insert_opr() fails.
     */
    virtual void on_opr_del(OperatorNode* opr) const noexcept = 0;

    //! update shapes of output vars from shapes of input vars
    virtual void infer_shape(OperatorNode* opr,
                             ExecutionContext& ctx) const = 0;

    //! evaluate the bias term (coeff[0]) according to the operator semantics
    virtual void eval_bias(OperatorNode* opr, ExecutionContext& ctx) const = 0;

    //! infer_shape() and then eval_bias()
    void infer_shape_eval_bias(OperatorNode* opr, ExecutionContext& ctx) const {
        infer_shape(opr, ctx);
        eval_bias(opr, ctx);
    }

    /*!
     * \brief accumulate #VarNodeExeCtx::jacobian for the input vars
     *
     * The implementation can store auxiliary information in
     * #VarNodeExeCtx::user_data to speedup future calls of compute_order_bias()
     */
    virtual void accum_inp_grad(OperatorNode* opr,
                                ExecutionContext& ctx) const = 0;

    //! compute #VarNodeExeCtx::cur_order_bias from input coeffs
    virtual void compute_order_bias(OperatorNode* opr,
                                    ExecutionContext& ctx) const = 0;

    //! compute #VarNodeExeCtx::coeff[k] from input coeffs and
    //! #VarNodeExeCtx::cur_order_bias
    virtual void compute_coeff(OperatorNode* opr,
                               ExecutionContext& ctx) const = 0;
};

//! a single operator instance
class OperatorNode : public NonCopyable {
    ComputingGraph* const m_cg;
    const OperatorMeta* const m_meta;
    const size_t m_id;
    void* m_storage;
    VarNodeArray m_inputs;
    size_t m_nr_outputs;
    std::unique_ptr<VarNode[]> m_outputs_storage;

    [[noreturn]] void on_visit_invalid_input(size_t i) const;
    [[noreturn]] void on_visit_invalid_output(size_t i) const;

public:
    OperatorNode(ComputingGraph* cg, const OperatorMeta* meta, void* storage,
                 VarNodeArray inputs);
    ~OperatorNode();

    ComputingGraph* owner_graph() const { return m_cg; }

    const OperatorMeta* meta() const { return m_meta; }

    //! check whether this operator is an instance of the given opr meta type
    template <typename Meta>
    bool isinstance() const {
        return meta() == Meta::instance();
    }

    //! opaque storage associated with this opr, used by OperatorMeta
    void* storage() const { return m_storage; }

    VarNode* input(size_t i) const {
        if (i >= m_inputs.size()) {
            on_visit_invalid_input(i);
        }
        return m_inputs[i];
    }

    const VarNodeArray& inputs() const { return m_inputs; }

    size_t nr_outputs() const { return m_nr_outputs; }

    VarNode* output(size_t i) const {
        if (i >= m_nr_outputs) {
            on_visit_invalid_output(i);
        }
        return m_outputs_storage.get() + i;
    }

    std::span<VarNode> outputs() const {
        return {m_outputs_storage.get(), m_nr_outputs};
    }

    std::string str() const;
};

/*!
 * \brief A symbolic representation of a pure function
 *
 * This class manages the memory of the operators and variables. Its lifespan
 * must be longer than the oprs and vars being used.
 */
class ComputingGraph {
    size_t m_obj_id = 0;
    std::vector<std::unique_ptr<OperatorNode>> m_oprs;

public:
    OperatorNode* insert_opr(const OperatorMeta* meta, void* storage,
                             VarNodeArray inputs);

    //! next object ID; object ID uniquely identifies an object in this graph
    size_t next_obj_id() { return m_obj_id++; }
};

//! return the operators in topological order that produce given variables
OprNodeArray topo_sort(const VarNodeArray& dest);

//! map from PlaceholderOprMeta outputs to corresponding values
class TensorValueMap {
    std::unordered_map<VarNode*, TensorND> m_map;

public:
    //! insert a new value, which must not duplicate previous vars
    TensorValueMap& insert(VarNode* var, TensorND value);

    //! get the value of a var
    const TensorND& get(VarNode* var) const { return m_map.at(var); }
    TensorND& get(VarNode* var) { return m_map.at(var); }

    //! pack the values in a specific order as specified by \p vars
    TensorArray pack(std::span<VarNode* const> vars) const;

    auto begin() { return m_map.begin(); }
    auto end() { return m_map.end(); }
    auto begin() const { return m_map.begin(); }
    auto end() const { return m_map.end(); }
};

/*!
 * \brief Computing the numerical coefficients in the Taylor expansion of the
 *      given function
 *
 * Specifically, given a function f(x) (represented as a VarNode*), assuming
 * that \f$ x(a)=\sum_i x_i (a-a_i)^i \f$, this class iteratively computes
 * \f$f_i\f$ such that \f$f(x(a)) = \sum_i f_i (a-a_i)^i\f$. For each \f$i\f$,
 * \f$k_i\f$ and \f$b_i\f$ are first computed such that \f$f_i=k_i x_i+b_i\f$,
 * and then the caller should solve \f$x_i\f$ using external constraints. In
 * fact, \f$k_i\f$ is the same for all \f$i\f$, which is the Jacobian.
 *
 * Basically, the user first calls push_xi() to initialize the solver at the
 * starting point, and then alternatively calls compute_next_order_bias() and
 * push_xi().
 *
 * Note that we use batched computing. The first dimension of all vars is the
 * batch size.
 */
class TaylorCoeffProp : public NonCopyable {
    friend class ParallelTaylorCoeffProp;
    const OprNodeArray m_topo_order;
    const bool m_output_is_batched;
    bool m_xi_known = false;
    bool m_jacobian_done = false;
    VarNodeArray m_input_vars;
    VarNode* m_output_var = nullptr;
    ExecutionContext m_exe_ctx;

    void ensure_jacobian();

public:
    TaylorCoeffProp(VarNode* output, bool output_is_batched);

    //! the input vars (i.e., vars produced by the PlaceholderOprMeta); their
    //! order decides the order of values in push_xi()
    const VarNodeArray& inputs() const { return m_input_vars; }

    /*!
     * \brief Push the value of \f$x_i\f$ solved by the user of this class.
     *
     * This function must be called before each call of
     * compute_next_order_bias(). The first call initializes the bias term.
     *
     * \return the coeff evaluated at the output node
     */
    const TensorND& push_xi(std::span<const TensorND> xi);

    const TensorND& push_xi(const TensorValueMap& vmap) {
        return push_xi(vmap.pack(inputs()));
    }

    //! overload of push_xi() in the case of a single input
    const TensorND& push_xi(const TensorND& xi) {
        return push_xi(std::span<const TensorND>{&xi, 1});
    }

    //! increase order and return the computed bias of the coefficient \f$b_i\f$
    const TensorND& compute_next_order_bias();

    //! get previous return value of compute_next_order_bias()
    const TensorND& get_prev_next_order_bias() const;

    //! get the Jacobian of a particular input
    const StSparseLinearTrans& get_jacobian(VarNode* x);
};

/*!
 * \brief a parallel wrapper of TaylorCoeffProp by splitting the batch
 *
 * This class implements data parallelism by splitting on the batch. It requires
 * there be only one input variable, and the output must be batched.
 */
class ParallelTaylorCoeffProp : public NonCopyable {
    class Impl;
    std::unique_ptr<Impl> m_pimpl;

public:
    //! see TaylorCoeffProp
    explicit ParallelTaylorCoeffProp(VarNode* var);
    ~ParallelTaylorCoeffProp();

    //! see TaylorCoeffProp::push_xi()
    ParallelTaylorCoeffProp& push_xi(const TensorND& val);

    //! see TaylorCoeffProp::compute_next_order_bias()
    const TensorND& compute_next_order_bias();

    //! get previous return value of compute_next_order_bias()
    const TensorND& get_prev_next_order_bias() const;

    //! see TaylorCoeffProp::get_jacobian
    StSparseLinearTrans get_jacobian();

    //! gather values of previous push_xi() call
    TensorND gather_yi() const;

    //! get number of workers
    size_t nr_worker() const;

    using WorkerTaskFn = std::function<void(size_t worker_id)>;

    //! execute a function on the worker threads (so the worker threads can be
    //! easily reused)
    void run_on_workers(const WorkerTaskFn& fn);
};

//! evaluate the numerical value of a unary function
TensorND eval_unary_func(VarNode* y, const TensorND& xval);

ComputingGraph* VarNode::owner_graph() const {
    return owner_opr()->owner_graph();
}

}  // namespace symbolic

using symbolic::TensorValueMap;
}  // namespace sanm
