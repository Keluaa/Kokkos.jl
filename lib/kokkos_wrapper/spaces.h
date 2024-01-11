
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


/**
 * Return a `TList<Match>{}` where `Match` is the space with the same name as 'filter'. There can only be only one match.
 * If there is no match, `TList<Default>{}` is returned instead.
 */
template<const std::string_view& filter, typename Default, typename... S>
auto find_space(TList<S...> spaces)
{
    auto matches = filter_types([&](auto space) {
        return std::bool_constant<SpaceInfo<decltype(space)>::julia_name == std::string_view(filter)>{};
    }, spaces);

    if constexpr (matches.size == 0) {
        return TList<Default>{};
    } else {
        static_assert(matches.size <= 1, "Several spaces have the same name");  // Should not happen
        return matches;
    }
}


void define_all_spaces(jlcxx::Module& mod);

void define_space_specific_methods(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_SPACES_H
