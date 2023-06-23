
#include "copy.h"

#include "views.h"
#include "memory_spaces.h"
#include "execution_spaces.h"
#include "utils.h"

#ifndef WITHOUT_EXEC_SPACE_ARG
#define WITHOUT_EXEC_SPACE_ARG 0
#endif


struct NoExecSpaceArg {};


template<typename ExecSpace, typename DestView>
struct DeepCopyDetector
{
    template<typename SrcView>
    using deep_copyable_t = decltype(Kokkos::deep_copy(ExecSpace{}, DestView{}, SrcView{}));

    template<typename SrcView>
    using is_deep_copyable = Kokkos::is_detected<deep_copyable_t, SrcView>;
};


template<typename DestView>
struct DeepCopyDetectorNoExecSpace
{
    template<typename SrcView>
    using deep_copyable_t = decltype(Kokkos::deep_copy(DestView{}, SrcView{}));

    template<typename SrcView>
    using is_deep_copyable = Kokkos::is_detected<deep_copyable_t, SrcView>;
};


void register_all_deep_copy_combinations(jlcxx::Module& mod)
{
    auto combinations = build_all_combinations<
#if COMPLETE_BUILD == 1
            decltype(TList<NoExecSpaceArg>{} + FilteredExecutionSpaceList{}),
#elif WITHOUT_EXEC_SPACE_ARG == 0
            decltype(FilteredExecutionSpaceList{}),
#else
            decltype(TList<NoExecSpaceArg>{}),
#endif
            decltype(tlist_from_sequence(DimensionsToInstantiate{})),
            TList<VIEW_TYPES>,
            DestLayoutList,
            DestMemSpaces
    >();

    auto src_combinations = build_all_combinations<
            LayoutList,
            FilteredMemorySpaceList
    >();

    apply_to_all(combinations, [&](auto combination)
    {
        using ExecSpace = typename decltype(combination)::template Arg<0>;
        using Dimension = typename decltype(combination)::template Arg<1>;
        using Type = typename decltype(combination)::template Arg<2>;

        using DestLayout = typename decltype(combination)::template Arg<3>;
        using DestMemSpace = typename decltype(combination)::template Arg<4>;

        using DestView = ViewWrap<Type, Dimension, DestLayout, DestMemSpace>;

        using Detector = std::conditional_t<std::is_same_v<ExecSpace, NoExecSpaceArg>,
                DeepCopyDetectorNoExecSpace<typename DestView::kokkos_view_t>,
                DeepCopyDetector<ExecSpace, typename DestView::kokkos_view_t>>;

        apply_to_all(src_combinations, [&](auto src_combination)
        {
            using SrcLayout = typename decltype(src_combination)::template Arg<0>;
            using SrcMemSpace = typename decltype(src_combination)::template Arg<1>;

            using SrcView = ViewWrap<Type, Dimension, SrcLayout, SrcMemSpace>;

            constexpr bool is_deep_copyable = Detector::template is_deep_copyable<typename SrcView::kokkos_view_t>::value;

            if constexpr (std::is_same_v<ExecSpace, NoExecSpaceArg>) {
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
            } else {
                mod.method("deep_copy",
                [](const ExecSpace& exec_space, const DestView& dest_view, const SrcView& src_view)
                {
                    if constexpr (is_deep_copyable) {
                        Kokkos::deep_copy(exec_space, dest_view, src_view);
                    } else {
                        jl_errorf("Deep copy is not possible from `%s` to `%s` in `%s`",
                                  jl_typename_str((jl_value_t*) jlcxx::julia_type<SrcView>()),
                                  jl_typename_str((jl_value_t*) jlcxx::julia_type<DestView>()),
                                  jl_typename_str((jl_value_t*) jlcxx::julia_type<ExecSpace>()->super->super));
                    }
                });
            }
        });
    });
}


#ifdef WRAPPER_BUILD
void define_kokkos_deep_copy(jlcxx::Module& mod)
{
    // Called from 'Kokkos.Wrapper.Impl'
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* views_module = (jl_module_t*) jl_get_global(wrapper_module->parent, jl_symbol("Views"));
#else
JLCXX_MODULE define_kokkos_module(jlcxx::Module& mod)
{
    // Called from 'Kokkos.Views.Impl<number>'
    jl_module_t* views_module = mod.julia_module()->parent;
#endif

    jl_module_import(mod.julia_module(), views_module, jl_symbol("deep_copy"));

    mod.set_override_module(views_module);
    register_all_deep_copy_combinations(mod);
    mod.unset_override_module();
}
