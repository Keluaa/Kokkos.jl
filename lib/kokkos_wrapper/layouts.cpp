
#include "layouts.h"
#include "utils.h"


template<typename Layout>
void register_layout(jl_module_t* views_module)
{
    std::string name;
    if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
        name = "LayoutLeft";
    } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutRight>) {
        name = "LayoutRight";
    } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutStride>) {
        name = "LayoutStride";
    } else {
        static_assert(std::is_same_v<Layout, void>, "Unknown layout type");
    }

    // Equivalent to 'mod.map_type<Layout>(name)', but by using a type defined in another module.

    auto* dt = (jl_datatype_t*) jlcxx::julia_type(name, views_module);
    if (dt == nullptr) {
        throw std::runtime_error("Type for " + name + " was not found when mapping it.");
    }
    jlcxx::set_julia_type<Layout>(dt);
}


template<template<typename> typename Container, typename... L>
void register_layouts(jl_module_t* views_module, Container<L...>)
{
    ([&](){
        static bool already_registered = false;
        if (!already_registered) {
            register_layout<L>(views_module);
            already_registered = true;
        }
    }(), ...);
}


constexpr auto view_layout_count = LayoutList::size;
using julia_layouts = std::array<jl_value_t*, view_layout_count>;

template<std::size_t... I, typename... T>
void add_julia_layouts(julia_layouts& array, std::integer_sequence<std::size_t, I...>, TList<T...>)
{
    ([&](){ array[I] = (jl_value_t*) jlcxx::julia_base_type<T>(); } (), ...);
}


julia_layouts build_julia_layouts_array()
{
    const std::make_index_sequence<view_layout_count> indexes{};
    julia_layouts array{};
    add_julia_layouts(array, indexes, LayoutList{});
    return array;
}


void define_all_layouts(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* views_module = (jl_module_t*) jl_get_global(wrapper_module->parent, jl_symbol("Views"));

    register_layouts(views_module, LayoutList{});
    mod.method("__compiled_layouts", [](){ return std::tuple_cat(build_julia_layouts_array()); });
}
