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
 * Template list where the N-th argument can be accessed with 'TList<...>::Arg<N>', and which can be combined with other
 * template lists.
 *
 * Will not work with reference types.
 */
template<typename... Args>
struct TList {
    template<std::size_t I>
    using Arg = std::remove_reference_t<decltype(std::get<I>(std::tuple<Args...>{}))>;

    static constexpr auto size = sizeof...(Args);

    template<typename... OtherArgs>
    constexpr TList<Args..., OtherArgs...> operator+(TList<OtherArgs...>) { return {}; }
};


template<template<typename> typename List, typename... Args>
constexpr TList<Args...> to_TList(List<Args...>)
{
    return {};
}


template<template<typename...> typename List, typename... Args>
constexpr jlcxx::ParameterList<Args...> to_parameter_list(List<Args...>)
{
    return jlcxx::ParameterList<Args...>{};
}


/**
 * Helper function to transform a 'std::integer_sequence<int, 1, 2, ...>' into a
 * 'jlcxx::ParameterList<std::integral_constant<std::size_t, 1>, ...>'
 * This way each dimension is stored into its own type, instead of a single sequence type.
 */
template<typename I, I... Dims>
constexpr auto wrap_dims(std::integer_sequence<I, Dims...>)
{
    return to_parameter_list((TList<std::integral_constant<std::size_t, Dims>>{} + ...));
}


template<typename Element, typename... Args>
constexpr bool is_element_unique(TList<Args...>)
{
    return !(std::is_same_v<Element, Args> || ...);
}


template<typename... Unique, typename Element, typename... Args>
constexpr auto remove_duplicates(TList<Unique...>, TList<Element, Args...>)
{
    if constexpr (sizeof...(Args) == 0) {
        return TList<Unique..., Element>{};
    } else if constexpr (is_element_unique<Element>(TList<Args...>{})) {
        return remove_duplicates(TList<Unique..., Element>{}, TList<Args...>{});
    } else {
        return remove_duplicates(TList<Unique...>{}, TList<Args...>{});
    }
}


/**
 * Simple recursive template function to return the template list with no duplicate types.
 */
template<template<typename> typename List, typename... Args>
constexpr auto remove_duplicates(List<Args...>)
{
    return remove_duplicates(TList<>{}, TList<Args...>{});
}


template<typename... Stack, typename... List>
constexpr auto build_all_combinations(TList<Stack...>, TList<List...>)
{
    return (TList<TList<Stack..., List>>{} + ...);
}


template<typename NextList, typename... NextLists, typename... Stack, typename... List>
constexpr auto build_all_combinations(TList<Stack...>, TList<List...>)
{
    if constexpr (sizeof...(NextLists) == 0) {
        return (build_all_combinations<>(TList<Stack..., List>{}, NextList{}) + ...);
    } else {
        return (build_all_combinations<NextLists...>(TList<Stack..., List>{}, NextList{})+ ...);
    }
}


/**
 * Return a TList containing other TLists of all combinations of all elements of all given lists.
 *
 * Example:
 *
 *     using floats = TList<double, float>;
 *     using ints = TList<short, int, long>;
 *     using combinations = build_all_combinations<floats, ints>();
 *
 * Then `combinations` is of type:
 *
 *     TList<
 *          TList<double, short>,
 *          TList<double, int>,
 *          TList<double, long>,
 *          TList<float, short>,
 *          TList<float, int>,
 *          TList<float, long>
 *     >
 */
template<typename List, typename... NextLists>
constexpr auto build_all_combinations()
{
    return build_all_combinations<NextLists...>(TList<>{}, List{});
}


template<typename Functor, typename... Combinations>
void apply_to_all(TList<Combinations...>, Functor&& f)
{
    (f(Combinations{}), ...);
}


#endif //KOKKOS_WRAPPER_UTILS_H
