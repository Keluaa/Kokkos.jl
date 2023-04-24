
#include "copy.h"

#include "views.h"
#include "memory_spaces.h"
#include "execution_spaces.h"


template<typename... Args>
struct MyDestView {
    template<std::size_t I>
    using Arg = std::remove_reference_t<decltype(std::get<I>(std::tuple<Args...>{}))>;
};


/**
 * Hack to use 'jlcxx::TypeWrapper<T>::apply_combination' outside of a 'jlcxx::Module::add_type' context
 */
struct MyApplyTypes {
    template<typename AppT, typename FunctorT>
    void apply(FunctorT&& func)
    {
        func(AppT{});
    }

    template<template<typename...> class TemplateT, typename... TypeLists, typename FunctorT>
    void apply_combination(FunctorT&& ftor)
    {
        this->template apply_combination<jlcxx::ApplyType<TemplateT>, TypeLists...>(std::forward<FunctorT>(ftor));
    }

    template<typename ApplyT, typename... TypeLists, typename FunctorT>
    void apply_combination(FunctorT&& ftor)
    {
        typedef typename jlcxx::CombineTypes<ApplyT, TypeLists...>::type applied_list;
        jlcxx::detail::DoApply<applied_list>()(*this, std::forward<FunctorT>(ftor));
    }
};


template<typename... Args>
struct FuseExpansionPackResult {
    template<typename A>
    FuseExpansionPackResult<Args..., A> operator,(FuseExpansionPackResult<A>)
    {
        return FuseExpansionPackResult<Args..., A>{};
    }

    using ParameterList = jlcxx::ParameterList<Args...>;
};


template<typename T>
FuseExpansionPackResult<T> fuse_pack(const T&)
{
    return FuseExpansionPackResult<T>{};
}


template<typename I, I... Dims>
auto wrap_dims(std::integer_sequence<I, Dims...>)
{
    return ([&](){ return fuse_pack(std::integral_constant<std::size_t, Dims>{}); }(), ...);
}


template<template<typename...> typename List, typename... Args>
jlcxx::ParameterList<Args...> to_parameter_list(List<Args...>)
{
    return jlcxx::ParameterList<Args...>{};
}


template<typename T, template<typename...> typename List, typename... Args>
jlcxx::ParameterList<T, Args...> to_parameter_list(List<Args...>)
{
    return jlcxx::ParameterList<T, Args...>{};
}


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
    // Transform a 'std::integer_sequence<int, 1, 2, ...>' into a 'jlcxx::ParameterList<std::integral_constant<std::size_t, 1>, ...>'
    // This way each dimension is stored into its own type, instead of a single sequence type.
    using DimsList = decltype(wrap_dims(DimensionsToInstantiate{}))::ParameterList;
    using MemSpacesList = decltype(to_parameter_list(MemorySpacesList{}));
    using ExecSpacesList = decltype(to_parameter_list<NoExecSpaceArg>(ExecutionSpaceList{}));

    MyApplyTypes{}.apply_combination<
            MyDestView,
            ExecSpacesList,
            DimsList,
            jlcxx::ParameterList<VIEW_TYPES>,
            MemSpacesList
        >(
    [&](auto wrapped)
    {
        using ExecSpace = typename decltype(wrapped)::template Arg<0>;
        using Dimension = typename decltype(wrapped)::template Arg<1>;
        using DestType = typename decltype(wrapped)::template Arg<2>;
        using DestMemSpace = typename decltype(wrapped)::template Arg<3>;

        using DestView = ViewWrap<DestType, Dimension, DestMemSpace>;

        using Detector = std::conditional_t<std::is_same_v<ExecSpace, NoExecSpaceArg>,
                DeepCopyDetectorNoExecSpace<typename DestView::kokkos_view_t>,
                DeepCopyDetector<ExecSpace, typename DestView::kokkos_view_t>>;

        MyApplyTypes{}.apply_combination<
                MyDestView,
                jlcxx::ParameterList<VIEW_TYPES>,
                MemSpacesList
            >(
        [&](auto src_view_info)
        {
            using SrcType = typename decltype(src_view_info)::template Arg<0>;
            using SrcMemSpace = typename decltype(src_view_info)::template Arg<1>;

            using SrcView = ViewWrap<SrcType, Dimension, SrcMemSpace>;

            if constexpr (Detector::template is_deep_copyable<typename SrcView::kokkos_view_t>::value) {
                if constexpr (std::is_same_v<ExecSpace, NoExecSpaceArg>) {
                    mod.method("deep_copy",
                    [](const DestView& dest_view, const SrcView& src_view)
                    {
                        Kokkos::deep_copy(dest_view, src_view);
                    });
                } else {
                    mod.method("deep_copy",
                    [](const ExecSpace& exec_space, const DestView& dest_view, const SrcView& src_view)
                    {
                        Kokkos::deep_copy(exec_space, dest_view, src_view);
                    });
                }
            }
        });
    });
}


void define_kokkos_deep_copy(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* views_module = (jl_module_t*) jl_get_global(wrapper_module->parent, jl_symbol("Views"));
    jl_module_import(mod.julia_module(), views_module, jl_symbol("deep_copy"));

    mod.set_override_module(views_module);
    register_all_deep_copy_combinations(mod);
    mod.unset_override_module();
}
