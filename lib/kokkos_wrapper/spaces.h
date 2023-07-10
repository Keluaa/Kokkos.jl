
#ifndef KOKKOS_WRAPPER_SPACES_H
#define KOKKOS_WRAPPER_SPACES_H

#include "jlcxx/jlcxx.hpp"
#include "utils.h"


/**
 * Info structs defined for each enabled execution or memory space.
 *
 * On the Julia side, each of these structs are mapped to an abstract type with the same name as `julia_name`.
 */
template<typename Space>
struct SpaceInfo {
    using space = void;
    static constexpr const std::string_view julia_name{};
};


template<std::size_t N, const std::array<const char*, N>& filters, typename... S, std::size_t... I>
auto filter_spaces(TList<S...> spaces, std::index_sequence<I...>)
{
    return filter_types([&](auto space) {
        return std::bool_constant<((SpaceInfo<decltype(space)>::julia_name == std::string_view(filters[I])) || ...)>{};
    }, spaces);
}


/**
 * From an array of `N` space names, filter `spaces` to include only those whose name match one of the filters.
 */
template<std::size_t N, const std::array<const char*, N>& filters, typename... S>
auto filter_spaces(TList<S...> spaces)
{
    if constexpr (N == 0 || sizeof...(S) == 0) {
        return spaces;
    } else {
        return filter_spaces<N, filters>(spaces, std::make_index_sequence<N>{});
    }
}


void define_all_spaces(jlcxx::Module& mod);

void define_space_specific_methods(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_SPACES_H
