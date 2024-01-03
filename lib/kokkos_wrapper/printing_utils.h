#ifndef KOKKOS_WRAPPER_PRINTING_UTILS_H
#define KOKKOS_WRAPPER_PRINTING_UTILS_H

#include <string>
#include <string_view>
#include <ostream>

#include "utils.h"


/**
 * Stolen from https://stackoverflow.com/a/20170989, with support for more compilers.
 */
template <typename T>
constexpr std::string_view get_type_name()
{
#if defined(__clang__)
    constexpr auto prefix = std::string_view{"[T = "};
    constexpr auto suffix = "]";
    constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(__INTEL_COMPILER) || defined(__NVCOMPILER)
    constexpr auto prefix = std::string_view{"with T = "};
    constexpr auto suffix = "]";
    constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
    constexpr auto prefix = std::string_view{"with T = "};
    constexpr auto suffix = "; ";
    constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
    constexpr auto prefix = std::string_view{"get_type_name<"};
    constexpr auto suffix = ">(void)";
    constexpr auto function = std::string_view{__FUNCSIG__};
#else
#error "Unsupported compiler. Please report it to https://github.com/Keluaa/Kokkos.jl alongside your compiler info."
#endif

    const auto start = function.find(prefix) + prefix.size();
    const auto end = function.find(suffix);
    const auto size = end - start;

    return function.substr(start, size);
}


template<typename... T>
std::ostream& type_to_string(std::ostream& os, const std::string& indent, const TList<T...>&)
{
    int i = 0;
    const auto elem_indent = indent + "    ";
    os << "TList<";
    ([&](){
        if constexpr (is_tlist_v<T>) {
            os << "\n" << elem_indent << "[" << i << "] ";
            type_to_string(os, elem_indent + "    ", T{});
        } else {
            os << "\n" << elem_indent << "[" << i << "] " << get_type_name<T>();
        }
        i++;
        if (i < sizeof...(T)) os << ", ";
    }(), ...);
    if (sizeof...(T) > 0)
        os << "\n" << indent;
    os << ">";
    return os;
}


template<typename... T>
std::ostream& operator<<(std::ostream& os, const TList<T...>& tlist)
{
    return type_to_string(os, "", tlist);
}


#endif //KOKKOS_WRAPPER_PRINTING_UTILS_H
