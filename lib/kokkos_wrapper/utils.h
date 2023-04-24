#ifndef KOKKOS_WRAPPER_UTILS_H
#define KOKKOS_WRAPPER_UTILS_H

#include "jlcxx/jlcxx.hpp"


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


/**
 * Template list where the N-th argument can be accessed with 'TList<...>::Arg<N>'.
 *
 * Will not work with reference types.
 */
template<typename... Args>
struct TList {
    template<std::size_t I>
    using Arg = std::remove_reference_t<decltype(std::get<I>(std::tuple<Args...>{}))>;
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


/**
 * Factory method to combine the resulting types of a template parameter pack expansion into one.
 */
template<typename T>
FuseExpansionPackResult<T> fuse_pack(const T&)
{
    return FuseExpansionPackResult<T>{};
}


/**
 * Helper function to transform a 'std::integer_sequence<int, 1, 2, ...>' into a
 * 'jlcxx::ParameterList<std::integral_constant<std::size_t, 1>, ...>'
 * This way each dimension is stored into its own type, instead of a single sequence type.
 */
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


#endif //KOKKOS_WRAPPER_UTILS_H
