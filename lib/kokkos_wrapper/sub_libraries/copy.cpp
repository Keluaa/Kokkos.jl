
#include "views.h"
#include "memory_spaces.h"
#include "execution_spaces.h"
#include "utils.h"


template<typename DstView, typename SrcView>
using deep_copyable_no_exec_t = decltype(Kokkos::deep_copy(DstView{}, SrcView{}));


template<typename ExecSpace, typename DstView, typename SrcView>
using deep_copyable_t = decltype(Kokkos::deep_copy(ExecSpace{}, DstView{}, SrcView{}));


template<
        typename Type, typename Dim,
        typename SrcLayout, typename SrcMemSpace,
        typename DstLayout, typename DstMemSpace>
void register_deep_copy_method_without_exec_space(jlcxx::Module& mod)
{
    using SrcView = ViewWrap<Type, Dim, SrcLayout, SrcMemSpace>;
    using DestView = ViewWrap<Type, Dim, DstLayout, DstMemSpace>;

    constexpr bool is_deep_copyable = Kokkos::is_detected<deep_copyable_no_exec_t, DestView, SrcView>::value;

    mod.method("deep_copy",
    [](const DestView& dest_view, const SrcView& src_view)
    {
        if constexpr (is_deep_copyable) {
            Kokkos::deep_copy(dest_view, src_view);
        } else {
            jl_errorf("Deep copy is not possible from `%s` to `%s`",
                      jl_typename_str((jl_value_t*) jlcxx::julia_type<SrcView>()),
                      jl_typename_str((jl_value_t*) jlcxx::julia_type<DestView>()));
        }
    });
}


template<
        typename Type, typename Dim,
        typename SrcLayout, typename SrcMemSpace,
        typename DstLayout, typename DstMemSpace,
        typename ExecSpace>
void register_deep_copy_method_with_exec_space(jlcxx::Module& mod)
{
    using SrcView = ViewWrap<Type, Dim, SrcLayout, SrcMemSpace>;
    using DestView = ViewWrap<Type, Dim, DstLayout, DstMemSpace>;

    constexpr bool is_deep_copyable = Kokkos::is_detected<deep_copyable_t, ExecSpace, DestView, SrcView>::value;

    mod.method("deep_copy",
    [](const ExecSpace& exec_space, const DestView& dest_view, const SrcView& src_view)
    {
        if constexpr (is_deep_copyable) {
            Kokkos::deep_copy(exec_space, dest_view, src_view);
        } else {
            jl_errorf("Deep copy is not possible from `%s` to `%s` in `%s`",
                      jl_typename_str((jl_value_t*) jlcxx::julia_type<SrcView>()),
                      jl_typename_str((jl_value_t*) jlcxx::julia_type<DestView>()),
                      jl_typename_str((jl_value_t*) jlcxx::julia_type<ExecutionSpace>()->super->super));
        }
    });
}


JLCXX_MODULE define_kokkos_module(jlcxx::Module& mod)
{
    // Called from 'Kokkos.Views.Impl<number>'
    jl_module_t* views_module = mod.julia_module()->parent;
    jl_module_import(mod.julia_module(), views_module, jl_symbol("deep_copy"));

    if constexpr (std::is_void_v<DestMemorySpace>) {
        jl_errorf("No memory space with the name '" AS_STR(DEST_MEM_SPACE) "' for the destination memory space.\n"
                  "Compilation parameters:\n%s", get_params_string());
    } else if constexpr (WITHOUT_EXEC_SPACE_ARG == 1) {
        register_deep_copy_method_without_exec_space<VIEW_TYPE, Dimension, Layout, MemorySpace, DestLayout, DestMemorySpace>(mod);
    } else if constexpr (std::is_void_v<ExecutionSpace>) {
        jl_errorf("No execution space with the name '" AS_STR(EXEC_SPACE) "'.\n"
                  "Compilation parameters:\n%s", get_params_string());
    } else {
        register_deep_copy_method_with_exec_space<VIEW_TYPE, Dimension, Layout, MemorySpace, DestLayout, DestMemorySpace, ExecutionSpace>(mod);
    }

    mod.method("params_string", get_params_string);
}
