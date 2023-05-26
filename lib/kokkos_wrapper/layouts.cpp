
#include "layouts.h"
#include "utils.h"


template<typename Layout>
void register_layout(jl_module_t* views_module)
{
    std::string name(layout_name<Layout>());
    // Equivalent to 'mod.map_type<Layout>(name)', but by using a type defined in another module.

    auto* dt = (jl_datatype_t*) jlcxx::julia_type(name, views_module);
    if (dt == nullptr) {
        throw std::runtime_error("Type for " + name + " was not found when mapping it.");
    }
    jlcxx::set_julia_type<Layout>(dt);
}


template<typename... L>
void register_layouts(jl_module_t* views_module, TList<L...>)
{
    (register_layout<L>(views_module), ...);
}


auto build_julia_layouts_tuple()
{
    constexpr size_t layout_count = LayoutList::size;
    std::array<jl_value_t*, layout_count> array{};

    apply_with_index(LayoutList{}, [&](auto type, size_t i) {
        using T = typename decltype(type)::template Arg<0>;
        array[i] = (jl_value_t*) jlcxx::julia_type<T>();
    });

    return std::tuple_cat(array);
}


void define_all_layouts(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* views_module = (jl_module_t*) jl_get_global(wrapper_module->parent, jl_symbol("Views"));

    register_layouts(views_module, LayoutList{});
    mod.method("__compiled_layouts", [](){ return build_julia_layouts_tuple(); });
}
