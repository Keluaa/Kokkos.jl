
#include "layouts.h"
#include "utils.h"


using LayoutList = TList<Kokkos::LayoutLeft, Kokkos::LayoutRight, Kokkos::LayoutStride>;


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

    size_t i = 0;
    apply_to_each(LayoutList{}, [&](auto type) {
        using T = typename decltype(type)::template Arg<0>;
        array[i++] = (jl_value_t*) jlcxx::julia_type<T>();
    });

    return std::tuple_cat(array);
}


void define_all_layouts(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* main_module = (jl_module_t*) wrapper_module->parent;

    register_layouts(main_module, LayoutList{});
    mod.method("__compiled_layouts", [](){ return build_julia_layouts_tuple(); });
}
