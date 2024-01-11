#ifndef KOKKOS_WRAPPER_LAYOUTS_H
#define KOKKOS_WRAPPER_LAYOUTS_H

#include "Kokkos_Core.hpp"
#include "jlcxx/jlcxx.hpp"
#include "utils.h"
#include "parameters.h"


namespace LayoutListHelper {
    // Simple namespace which allows to reliably specify layouts from a macro without using their complete name

    using left = Kokkos::LayoutLeft;
    using right = Kokkos::LayoutRight;
    using stride = Kokkos::LayoutStride;
    using deviceDefault = Kokkos::DefaultExecutionSpace::array_layout;
    using hostDefault = Kokkos::DefaultHostExecutionSpace::array_layout;
    using NONE = void;

    using Layout = VIEW_LAYOUT;
    using DestLayout = DEST_LAYOUT;
}

using LayoutListHelper::Layout;
using LayoutListHelper::DestLayout;


template<typename Layout>
constexpr std::string_view layout_name()
{
    if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
        return "LayoutLeft";
    } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutRight>) {
        return "LayoutRight";
    } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutStride>) {
        return "LayoutStride";
    } else {
        static_assert(std::is_same_v<Layout, void>, "Unknown layout type");
        return "";
    }
}


// CxxWrap does not detect the layout types as simple types since they are classes. This is important as
// 'jlcxx::julia_base_type<Layout>' and 'jlcxx::SingletonType<Layout>' will return their supertype (Kokkos.Layout) if
// they are not marked as mirrored types.
template<>
struct jlcxx::IsMirroredType<Kokkos::LayoutLeft> : std::true_type {};
template<>
struct jlcxx::IsMirroredType<Kokkos::LayoutRight> : std::true_type {};
template<>
struct jlcxx::IsMirroredType<Kokkos::LayoutStride> : std::true_type {};


void define_all_layouts(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_LAYOUTS_H
