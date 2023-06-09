
#ifndef KOKKOS_WRAPPER_SPACES_H
#define KOKKOS_WRAPPER_SPACES_H

#include "jlcxx/jlcxx.hpp"


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


void define_all_spaces(jlcxx::Module& mod);

void define_space_specific_methods(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_SPACES_H
