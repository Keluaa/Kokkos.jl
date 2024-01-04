#ifndef KOKKOS_WRAPPER_UTILS_H
#define KOKKOS_WRAPPER_UTILS_H

#include <tuple>


#define AS_STR_IMPL(x) #x
#define AS_STR(x) AS_STR_IMPL(x)


/**
 * Template list where the N-th argument can be accessed with 'TList<...>::Arg<N>', and which can be combined with other
 * template lists.
 */
template<typename... Args>
struct TList {
    using tuple = std::tuple<Args...>;

    template<std::size_t I>
    using Arg = std::tuple_element_t<I, tuple>;

    static constexpr auto size = sizeof...(Args);

    template<typename... OtherArgs>
    constexpr TList<Args..., OtherArgs...> operator+(TList<OtherArgs...>) { return {}; }
};


template <typename>
struct is_tlist : public std::false_type {};

template <typename... T>
struct is_tlist<TList<T...>> : public std::true_type {};

template <typename T>
constexpr bool is_tlist_v = is_tlist<T>::value;


template<typename Functor, typename... Combinations, typename... Args>
void apply_to_each(TList<Combinations...>, Functor&& f, Args... args)
{
    (f(TList<Combinations>{}, std::forward<Args...>(args)...), ...);
}


template<typename T, std::size_t... I>
auto repeat_type(std::index_sequence<I...>)
{
    return ((I, TList<T>{}) + ...);  // Behold! A situation where the comma operator is required and useful!
}


/**
 * Returns a TList containing `N` copies of the type `T`, without recursion.
 */
template<typename T, std::size_t N>
auto repeat_type()
{
    if constexpr (N == 0) {
        return TList<>{};
    } else if constexpr (N == 1) {
        return TList<T>{};
    } else {
        return repeat_type<T>(std::make_index_sequence<N>{});
    }
}


/**
 * Returns a TList for which `f(T)` returns `std::true_type` (aka `std::bool_constant<true>`).
 */
template<typename Functor, typename... T>
constexpr auto filter_types(Functor&& f, TList<T...>)
{
    return (std::conditional_t<decltype(f(std::declval<T>()))::value, TList<T>, TList<>>{} + ...);
}


template<typename T, typename... V>
constexpr std::size_t count_same()
{
    return (std::is_same_v<T, V> + ...);
}


#endif //KOKKOS_WRAPPER_UTILS_H
