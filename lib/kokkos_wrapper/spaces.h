
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


JLCXX_MODULE define_all_spaces(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_SPACES_H
