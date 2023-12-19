
#include "memory_spaces.h"
#include "views.h"
#include "utils.h"


struct Nothing_t {};  // Mapped to `Core.Nothing`


void register_mirror_methods(jlcxx::Module& mod)
{
    using DstMemSpacesList = decltype(
            DestMemSpaces{}
#if WITH_NOTHING_ARG == 1
            + TList<Nothing_t>{}
#endif  // WITH_NOTHING_ARG == 1
    );

    auto combinations = build_all_combinations<
            FilteredMemorySpaceList,
            DstMemSpacesList
    >();

    if (!jlcxx::has_julia_type<Nothing_t>()) {
        jlcxx::set_julia_type<Nothing_t>(jl_nothing_type);
    }

    apply_to_all(combinations, [&](auto combination_list)
    {
        using SrcMemSpace = typename decltype(combination_list)::template Arg<0>;
        using DstMemSpace = typename decltype(combination_list)::template Arg<1>;

        using SrcView = ViewWrap<VIEW_TYPE, Dimension, Layout, SrcMemSpace>;

        mod.method("create_mirror", [](const SrcView& src_view, const DstMemSpace& dst_space, bool init)
        {
            if constexpr (std::is_same_v<DstMemSpace, Nothing_t>) {
                if (init) {
                    auto view_mirror = Kokkos::create_mirror(src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing, src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                }
            } else {
                if (init) {
                    auto view_mirror = Kokkos::create_mirror(dst_space, src_view);
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing, dst_space, src_view);
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                }
            }
        });

        mod.method("create_mirror_view", [](const SrcView& src_view, const DstMemSpace& dst_space, bool init)
        {
            if constexpr (std::is_same_v<DstMemSpace, Nothing_t>) {
                if (init) {
                    auto view_mirror = Kokkos::create_mirror_view(src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, src_view);
                    using default_dst_space = typename decltype(view_mirror)::memory_space;
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, default_dst_space>(std::move(view_mirror));
                }
            } else {
                if (init) {
                    auto view_mirror = Kokkos::create_mirror_view(dst_space, src_view);
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                } else {
                    auto view_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, dst_space, src_view);
                    return ViewWrap<VIEW_TYPE, Dimension, Layout, DstMemSpace>(std::move(view_mirror));
                }
            }
        });
    });
}


JLCXX_MODULE define_kokkos_module(jlcxx::Module& mod)
{
    // Called from 'Kokkos.Views.Impl<number>'
    jl_module_t* views_module = mod.julia_module()->parent;
    jl_module_import(mod.julia_module(), views_module, jl_symbol("create_mirror"));
    jl_module_import(mod.julia_module(), views_module, jl_symbol("create_mirror_view"));
    register_mirror_methods(mod);
    mod.method("params_string", get_params_string);
}
