
#include "memory_spaces.h"
#include "views.h"
#include "utils.h"


struct Nothing_t {};  // Mapped to `Core.Nothing`


template<typename SrcView, typename DstMemSpace>
void register_mirror_methods_with_dest_space(jlcxx::Module& mod)
{
    using MirrorView = typename SrcView::template with_mem_space<DstMemSpace>;

    mod.method("create_mirror",
    [](const SrcView& src_view, const DstMemSpace& dst_space, bool init)
    {
        if (init) {
            auto view_mirror = Kokkos::create_mirror(dst_space, src_view);
            return MirrorView(std::move(view_mirror));
        } else {
            auto view_mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing, dst_space, src_view);
            return MirrorView(std::move(view_mirror));
        }
    });

    mod.method("create_mirror_view",
    [](const SrcView& src_view, const DstMemSpace& dst_space, bool init) {
        if (init) {
            auto view_mirror = Kokkos::create_mirror_view(dst_space, src_view);
            return MirrorView(std::move(view_mirror));
        }
        else {
            auto view_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, dst_space, src_view);
            return MirrorView(std::move(view_mirror));
        }
    });
}


template<typename SrcView>
void register_mirror_methods_default_dest_space(jlcxx::Module& mod)
{
    mod.method("create_mirror", [](const SrcView& src_view, const Nothing_t&, bool init)
    {
        if (init) {
            auto view_mirror = Kokkos::create_mirror(src_view);
            using default_dst_space = typename decltype(view_mirror)::memory_space;
            return typename SrcView::template with_mem_space<default_dst_space>(std::move(view_mirror));
        } else {
            auto view_mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing, src_view);
            using default_dst_space = typename decltype(view_mirror)::memory_space;
            return typename SrcView::template with_mem_space<default_dst_space>(std::move(view_mirror));
        }
    });

    mod.method("create_mirror_view", [](const SrcView& src_view, const Nothing_t&, bool init) {
        if (init) {
            auto view_mirror = Kokkos::create_mirror_view(src_view);
            using default_dst_space = typename decltype(view_mirror)::memory_space;
            return typename SrcView::template with_mem_space<default_dst_space>(std::move(view_mirror));
        }
        else {
            auto view_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, src_view);
            using default_dst_space = typename decltype(view_mirror)::memory_space;
            return typename SrcView::template with_mem_space<default_dst_space>(std::move(view_mirror));
        }
    });
}


JLCXX_MODULE define_kokkos_module(jlcxx::Module& mod)
{
    // Called from 'Kokkos.Views.Impl<number>'
    jl_module_t* views_module = mod.julia_module()->parent;
    jl_module_import(mod.julia_module(), views_module, jl_symbol("create_mirror"));
    jl_module_import(mod.julia_module(), views_module, jl_symbol("create_mirror_view"));

    if (!jlcxx::has_julia_type<Nothing_t>()) {
        jlcxx::set_julia_type<Nothing_t>(jl_nothing_type);
    }

    if constexpr (std::is_void_v<MemorySpace>) {
        jl_errorf("No memory space with the name '" AS_STR(MEM_SPACE) "'\n"
                  "Compilation parameters:\n%s", get_params_string());
    } else if constexpr (WITH_NOTHING_ARG == 1) {
        using SrcView = ViewWrap<VIEW_TYPE, Dimension, Layout, MemorySpace>;
        register_mirror_methods_default_dest_space<SrcView>(mod);
    } else if constexpr (std::is_void_v<DestMemorySpace>) {
        jl_errorf("No memory space with the name '" AS_STR(DEST_MEM_SPACE) "' for destination space\n"
                  "Compilation parameters:\n%s", get_params_string());
    } else {
        using SrcView = ViewWrap<VIEW_TYPE, Dimension, Layout, MemorySpace>;
        register_mirror_methods_with_dest_space<SrcView, DestMemorySpace>(mod);
    }

    mod.method("params_string", get_params_string);
}
