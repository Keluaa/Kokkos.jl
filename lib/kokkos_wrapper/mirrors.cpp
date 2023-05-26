
#include "mirrors.h"

#include "memory_spaces.h"
#include "views.h"
#include "utils.h"


void register_mirror_methods(jlcxx::Module& mod)
{
    using DimsList = decltype(tlist_from_sequence(DimensionsToInstantiate{}));
    using DstMemSpacesList = decltype(TList<jl_value_t*>{} + MemorySpacesList{});

    auto combinations = build_all_combinations<
            DimsList,
            TList<VIEW_TYPES>,
            LayoutList,
            MemorySpacesList,
            DstMemSpacesList
    >();

    apply_to_all(combinations, [&](auto combination_list)
    {
        using Dimension = typename decltype(combination_list)::template Arg<0>;
        using Type = typename decltype(combination_list)::template Arg<1>;
        using Layout = typename decltype(combination_list)::template Arg<2>;
        using SrcMemSpace = typename decltype(combination_list)::template Arg<3>;
        using DstMemSpace = typename decltype(combination_list)::template Arg<4>;

        using SrcView = ViewWrap<Type, Dimension, Layout, SrcMemSpace>;

        mod.method("create_mirror", [](const SrcView& src_view, const DstMemSpace& dst_space, bool init)
        {
            if constexpr (std::is_same_v<DstMemSpace, jl_value_t*>) {
                if (!jl_is_nothing(dst_space)) {
                    jl_type_error("create_mirror", jl_nothing, dst_space);
                }
                if (init) {
                    auto view_mirror = Kokkos::create_mirror(src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<Type, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing, src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<Type, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                }
            } else {
                if (init) {
                    auto view_mirror = Kokkos::create_mirror(dst_space, src_view);
                    return ViewWrap<Type, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing, dst_space, src_view);
                    return ViewWrap<Type, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                }
            }
        });

        mod.method("create_mirror_view", [](const SrcView& src_view, const DstMemSpace& dst_space, bool init)
        {
            if constexpr (std::is_same_v<DstMemSpace, jl_value_t*>) {
                if (!jl_is_nothing(dst_space)) {
                    jl_type_error("create_mirror_view", jl_nothing, dst_space);
                }
                if (init) {
                    auto view_mirror = Kokkos::create_mirror_view(src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<Type, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<Type, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                }
            } else {
                if (init) {
                    auto view_mirror = Kokkos::create_mirror_view(dst_space, src_view);
                    return ViewWrap<Type, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, dst_space, src_view);
                    return ViewWrap<Type, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                }
            }
        });
    });
}


void define_kokkos_mirrors(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* views_module = (jl_module_t*) jl_get_global(wrapper_module->parent, jl_symbol("Views"));
    jl_module_import(mod.julia_module(), views_module, jl_symbol("create_mirror"));
    jl_module_import(mod.julia_module(), views_module, jl_symbol("create_mirror_view"));

    mod.set_override_module(views_module);
    register_mirror_methods(mod);
    mod.unset_override_module();
}
