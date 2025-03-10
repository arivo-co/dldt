/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once

#include <set>

#include "api/CPP/primitive.hpp"
#include "internal_primitive.h"

#include "meta_utils.h"

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

namespace cldnn
{

struct program_impl;
class reorder_inputs;
class graph_initializations;

template <class T>
struct typed_program_node;

template <class PType>
struct internal_primitive_type_base;

class json_composite;
class xml_composite;

/*
    Base class for all primitives which wraps API class and extends it to be used
    in graph context.

    Besides primitive description provided by user, this class includes functionality to
    ask for direct predecessors and succesors as well as takes care of changes to primitive
    which would affect other graph's nodes (the most commont case is probably calculating output layout).

    At graph level, all connections between nodes are directly stored inside program_nodes - in oposite
    to API level where all primitives store only ids of related ones.
*/
struct program_node
{
    friend struct program_impl;                     // to be removed when possible
    friend class compile_graph;                     // to be removed when possible
    friend class graph_initializations;             // to be removed when possible
    friend class prepare_primitive_fusing;          // to be removed when possible
    friend class prepare_conv_eltw_fusing;          // to be removed when possible
    friend class prepare_conv_eltw_read_write_opt;  // to be removed when possible
    friend class propagate_constants;               // to be removed when possible
    friend class post_optimize_weights;             // to be removed when possible - requires an access to selected_impl

    template <class PType>
    friend struct typed_program_node;

    program_node(std::shared_ptr<primitive> prim, program_impl& prog);

    program_node(program_node const&) = delete;

    virtual ~program_node() = default;

public:
    virtual const primitive_id& id() const { return desc->id; }
    virtual primitive_type_id type() const { return desc->type; }

    template <class PType>
    bool is_type() const
    {
        static_assert(meta::is_primitive<PType>::value, "Type argument for program_node::is_type should be a non-const, non-volatile type derived from primitive");
        return type() == PType::type_id();
    }

    program_impl& get_program() { return myprog; }
    program_impl const& get_program() const { return myprog; }

    std::shared_ptr<primitive_impl> get_selected_impl() const { return selected_impl; }

    std::vector<program_node*> const& get_dependencies() const { return dependencies; }
    program_node& get_dependency(size_t idx) const { return *dependencies.at(idx); }

    //replaces idx-th dependency of 'this' with 'new_dep', calls program::remove_if_dangling(old_dep)
    void replace_dependency(size_t idx, program_node& new_dep);
    //searches for 'old_dep' in dependencies list of 'this' and replaces it with 'new_dep', calls program::remove_if_dangling(old_dep)
    void replace_dependency(program_node const& old_dep, program_node& new_dep);

    std::vector<primitive_id> get_dependencies_ids() const;

    void remove_dependency(size_t idx);
    void remove_dependency(program_node& node);

    std::set<primitive_id> get_memory_dependencies() const;
    void add_memory_dependency(primitive_id);
    void add_memory_dependency(std::vector<primitive_id>);

    template<class PType>
    bool have_user_with_type() const
    {
        for (auto const& usr : users)
        {
            if (usr->is_type<PType>()) return true;
        }
        return false;
    }

    bool is_detached(bool whole_branch = false);

    std::list<program_node*> const& get_users() { return users; }
    // for const method, add const to stored successors/predecessors
    std::list<const program_node*> const& get_users() const { return reinterpret_cast<const std::list<const program_node*>&>(users); }

    std::unique_ptr<json_composite> desc_to_json() const;
    //do not modify primitive directly to keep synchronisation with graph
    std::shared_ptr<const primitive> get_primitive() const { return desc; }
    //primitive modification functions
    void set_output_padding(padding const& padd)
    {
        //changing output padding shouldn't cause any changes to other primitives
        //so just change it
        output_layout.data_padding = padd;
    }

    void merge_output_padding(padding const& padd)
    {
        set_output_padding(padding::max(padd, output_layout.data_padding));
    }

    //only calculated output layout (for external usage), does not modify/use cached output layout nor invalidate users
    layout calc_output_layout() const;

    //uses cached output layout if valid, if not calls 'calc_output_layout' and stores its result + invalidate all users if layout has changed and @p invalidate_users_if_changed is set to true
    layout get_output_layout(bool invalidate_users_if_changed = true);
    //returns cached output layout if valid, otherwise throws an exception
    layout get_output_layout() const;
    //returns result of get_output_layout without padding
    layout get_non_padded_output_layout(bool invalidate_users_if_changed = true);

    //sets cached output layout to an arbitrary value, invalidates users if new layout differs from previous one and @p invalidate_users_if_changed is set to true
    //returns whether output layout has changed
    bool set_output_layout(layout new_layout, bool invalidate_users_if_changed = true);

    //forces recalculation of cached output layout, invalidates users if new layout is different than previous one and @p invalidate_users_if_changed is set to true
    //returns whether output layout has changed
    bool recalc_output_layout(bool invalidate_users_if_changed = true);

    bool is_padded() { return static_cast<bool>(get_output_layout().data_padding); }
    bool is_padded() const { return static_cast<bool>(get_output_layout().data_padding); }

    bool has_padded_dependency();
    bool has_padded_dependency() const;

    bool is_input() const { return dependencies.empty(); }
    bool is_endpoint() const { return users.empty(); }
    void set_output(bool out) { output = out; }
    bool is_output() const { return output; }

    bool is_valid_output_layout() const { return valid_output_layout; }

    uint8_t mark(uint8_t val = 1) { uint8_t ret = user_mark; user_mark = val; return ret; }
    void unmark() { user_mark = 0; }
    bool is_marked() const { return user_mark != 0; }
    bool is_marked(uint8_t val) const { return user_mark == val; }
    uint8_t get_user_mark() const { return user_mark; }

    void set_fused_activation(cldnn_activation_func activation_func, cldnn_activation_additional_params additional_params)
    {
        fused_activation.activation_func = activation_func;
        fused_activation.additional_params = additional_params;
    }

    cldnn_activation_func get_fused_activation_func() const
    {
        return fused_activation.activation_func;
    }

    cldnn_activation_additional_params get_fused_activation_params() const
    {
        return fused_activation.additional_params;
    }

    // check/set if the node can be optimized out (removed from the network)
    bool can_be_optimized() const { return optimized; }
    void can_be_optimized(bool opt) { optimized = opt; }

    // check/set if the node's buffer can be shared during the memory pool optimization
    bool can_share_buffer() const { return share_buffer; }
    void can_share_buffer(bool share) { share_buffer = share; }

    // check/set if the node support padding in x,y,b and f
    bool support_padding() const { return _support_padding; }
    void support_padding(bool support) { _support_padding = support; }

    primitive_id get_org_primitive_id() const { return org_id; }

    bool is_constant() const { return constant; }
    
    // returns true if this node is within main data flow of the network (i.e. it does not describe helper data like convolution's weights etc.)
    bool is_in_data_flow() const { return data_flow; }

    //conversion from generic to specific
    template <class To, class..., class = typename std::enable_if<!std::is_same<To, primitive>::value>::type>
    typed_program_node<To>& as()
    {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To>&>(*this);
    }

    template <class To, class..., class = typename std::enable_if<!std::is_same<To, primitive>::value>::type>
    typed_program_node<To> const& as() const
    {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To> const&>(*this);
    }

    template <class To>
    operator typed_program_node<To>& ()
    {
        return as<To>();
    }

    template <class To>
    operator typed_program_node<To> const& () const
    {
        return as<To>();
    }

    void set_reused_memory_color(uint32_t color) const
    {
        has_reused_memory = true;
        reused_memory_color = color;
    }

    bool is_reusing_memory() { return has_reused_memory; };
    uint32_t get_reused_memory_color() { return reused_memory_color; ; }

protected:
    std::shared_ptr<primitive> desc;
    program_impl& myprog;

    std::shared_ptr<primitive_impl> selected_impl;

    bool valid_output_layout = false;
    layout output_layout = layout(data_types::f32, format::bfyx, tensor());

    std::vector<program_node*> dependencies;
    std::list<program_node*> users;

    // list of primitives that can reuse same memory buffers due to execution order conflicts
    std::set<primitive_id> memory_dependencies;

    bool constant = false;
    bool data_flow = false;

    bool output = false;
    uint8_t user_mark = 0;
    bool optimized = false;
    bool share_buffer = true;
    bool _support_padding = false;

    mutable bool has_reused_memory = false;
    mutable uint32_t reused_memory_color = 0;

    const primitive_id org_id;

    struct fused_activation_params
    {
        cldnn_activation_func activation_func = activation_none;
        cldnn_activation_additional_params additional_params = { 0.0f, 0.0f };
    };

    fused_activation_params fused_activation;

    void invalidate_users() const;
};

namespace details
{
    template <class PType>
    struct api_typed_program_node_base : public program_node
    {
        static_assert(meta::is_api_primitive<PType>::value, "PType should name a non-const, non-volatile type derived from cldnn::primitive but not from cldnn::internal_primitive");
        friend class cldnn::graph_initializations;
        friend struct cldnn::program_impl;
        friend class cldnn::reorder_inputs;
    public:
        using program_node::program_node;

        std::shared_ptr<const PType> get_primitive() const { return std::static_pointer_cast<const PType>(program_node::get_primitive()); }

    protected:
        std::shared_ptr<PType> typed_desc() const { return std::static_pointer_cast<PType>(desc); }
    };

    struct internal_program_node_base : public program_node
    {
        friend struct cldnn::program_impl;

        internal_program_node_base(program_impl& prog);

        const primitive_id& id() const override { return internal_id; }

        void set_implementation(std::unique_ptr<primitive_impl>&& impl);

    private:
        primitive_id internal_id;

        static primitive_id get_next_internal_id();
    };

    template <class PType>
    struct internal_typed_program_node_base : public internal_program_node_base
    {
        static_assert(meta::is_internal_primitive<PType>::value, "PType should name a non-const, non-volatile type derived from cldnn::internal_primitive");

    public:
        using internal_program_node_base::internal_program_node_base;

        primitive_type_id type() const override { return PType::type_id(); }

        template <class... Guard>
        [[noreturn]]
        void get_primitive(Guard&&...)
        {
            static_assert(meta::always_false<meta::pack<Guard...>>::value, "Trying to get primitive from internal node");
        }


    protected:
        template <class... Guard>
        [[noreturn]]
        void typed_desc(Guard&&...)
        {
            static_assert(meta::always_false<meta::pack<Guard...>>::value, "Trying to get primitive from internal node");
        }
    };
}

/*
Template class used to indicate that usage context requires 'program_node' to wrap primitive
of type 'PType'. Successful conversion from 'program_node' to 'typed_program_node<PType>' means
that this restriction in fact holds and functions/method/etc. may saftly use uderlaying primitive.

This class shadows 'get_primitive' method from base class which now returns pointer to more specific
type.
*/
template <class PType>
using typed_program_node_base = typename std::conditional<meta::is_api_primitive<PType>::value, details::api_typed_program_node_base<PType>, details::internal_typed_program_node_base<PType>>::type;

/*
    Actual template class used in context which requires 'program_node' to wrap
    primitive of type 'PType'. This class is introduced to provide possibility of explicit specialization.
    In most cases such specializations would add accessors to make access to PType-specific fields easier.

    It's not required to specialize this class for new primitives types.
*/
template <class PType>
struct typed_program_node : public typed_program_node_base<PType>
{
    using typed_program_node_base<PType>::typed_program_node_base;

    program_node& input() const { return program_node::get_dependency(0); }
};

}