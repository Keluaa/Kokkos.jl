#ifndef KOKKOS_WRAPPER_UTILS_H
#define KOKKOS_WRAPPER_UTILS_H

#include <tuple>


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


/**
 * Helper function to transform a 'std::integer_sequence<int, 1, 2, ...>' into a
 * 'TList<std::integral_constant<std::size_t, 1>, ...>'
 * This way each int is stored into its own type, instead of a single sequence type.
 */
template<typename I, I... Dims>
constexpr auto tlist_from_sequence(std::integer_sequence<I, Dims...>)
{
    if constexpr (sizeof...(Dims) == 0) {
        return TList<>{};
    } else {
        return (TList<std::integral_constant<std::size_t, Dims>>{} + ...);
    }
}

template<typename Element, typename... Args>
constexpr bool is_element_in_list(TList<Args...>)
{
    return (std::is_same_v<Element, Args> || ...);
}


template<typename... Unique, typename Element, typename... Args>
constexpr auto remove_duplicates(TList<Unique...>, TList<Element, Args...>)
{
    // If Element is present in the remaining Args, do not add it to the stack, then recurse with the remaining Args
    if constexpr (sizeof...(Args) == 0) {
        return TList<Unique..., Element>{};
    } else if constexpr (!is_element_in_list<Element>(TList<Args...>{})) {
        return remove_duplicates(TList<Unique..., Element>{}, TList<Args...>{});
    } else {
        return remove_duplicates(TList<Unique...>{}, TList<Args...>{});
    }
}


/**
 * Simple recursive template function to return the template list with no duplicate types.
 */
template<typename... Args>
constexpr auto remove_duplicates(TList<Args...>)
{
    if constexpr (sizeof...(Args) == 0) {
        return TList<>{};
    } else {
        return remove_duplicates(TList<>{}, TList<Args...>{});
    }
}


template<std::size_t offset, std::size_t... Idx>
auto offset_index_sequence(std::index_sequence<Idx...>)
{
    return std::index_sequence<(Idx + offset)...>{};
}


template<std::size_t start, std::size_t end>
auto build_index_sequence()
{
    static_assert(start >= 0);
    static_assert(start <= end);
    return offset_index_sequence<start>(std::make_index_sequence<end - start>{});
}


template<typename... T, std::size_t... I>
auto sub_tlist(TList<T...>, std::index_sequence<I...>)
{
    return TList<typename TList<T...>::template Arg<I>...>{};
}


/**
 * Returns the TList elements from 'start' to 'end' (exclusive).
 *
 *     sub_tlist<1, 3>(TList<short, int, long, double>) => TList<int, long>
 *     sub_tlist<1, 1>(TList<short, int, long, double>) => TList<>
 */
template<std::size_t start, std::size_t end, typename... T>
auto sub_tlist(TList<T...> l)
{
    static_assert(end <= sizeof...(T));
    return sub_tlist(l, build_index_sequence<start, end>());
}


template<typename... Stack, typename... List>
constexpr auto build_all_combinations(TList<Stack...>, TList<List...>)
{
    return (TList<TList<Stack..., List>>{} + ...);
}


template<typename NextList, typename... NextLists, typename... Stack, typename... List>
constexpr auto build_all_combinations(TList<Stack...>, TList<List...>)
{
    // Add to N copies of the stack one of the N elements of List, then recurse to the next lists (if there is any)
    if constexpr (sizeof...(NextLists) == 0) {
        if constexpr (sizeof...(List) == 0) {
            return build_all_combinations<>(TList<Stack...>{}, NextList{});
        } else {
            return (build_all_combinations<>(TList<Stack..., List>{}, NextList{}) + ...);
        }
    } else {
        if constexpr (sizeof...(List) == 0) {
            return build_all_combinations<NextLists...>(TList<Stack...>{}, NextList{});
        } else {
            return (build_all_combinations<NextLists...>(TList<Stack..., List>{}, NextList{}) + ...);
        }
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


template<typename Functor, typename... Combinations, typename... Args>
void apply_to_each(TList<Combinations...>, Functor&& f, Args... args)
{
    (f(TList<Combinations>{}, std::forward<Args...>(args)...), ...);
}


template<typename Functor, typename... Combinations, std::size_t... I, typename... Args>
void apply_with_index(TList<Combinations...>, std::index_sequence<I...>, Functor&& f, Args... args)
{
    (f(TList<Combinations>{}, I, std::forward<Args...>(args)...), ...);
}


/**
 * Applies `f` on each element of `combinations` (passed to `f` in a TList singleton) alongside their index.
 * `f` is called as `f(TList<Element>{}, i, args...)`.
 */
template<typename Functor, typename... Combinations, typename... Args>
void apply_with_index(TList<Combinations...> combinations, Functor&& f, Args... args)
{
    apply_with_index(combinations, std::index_sequence_for<Combinations...>{}, f, std::forward<Args...>(args)...);
}


template<typename T>
TList<T> repeat_type(std::size_t) { return {}; }


template<typename T, std::size_t... I>
auto repeat_type(std::index_sequence<I...>)
{
    return (repeat_type<T>(I) + ...);
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


#endif //KOKKOS_WRAPPER_UTILS_H
