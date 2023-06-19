#ifndef KOKKOS_WRAPPER_LAYOUTS_H
#define KOKKOS_WRAPPER_LAYOUTS_H

#include "Kokkos_Core.hpp"
#include "jlcxx/jlcxx.hpp"
#include "utils.h"

#ifndef WRAPPER_BUILD
#include "parameters.h"
#endif


#ifndef VIEW_LAYOUTS
/**
 * Controls which `Kokkos::View` layout types are to be instantiated.
 *
 * Layout types are specified as comma separated list of either:
 *  - a complete name: `Kokkos::LayoutRight` or `Kokkos::DefaultExecutionSpace::array_layout`
 *  - one of 'left', 'right', 'stride', 'deviceDefault' or 'hostDefault'
 *
 * Duplicates are allowed.
 *
 * The registered method `compiled_layouts` returns a tuple of all compiled layout types.
 */
#define VIEW_LAYOUTS deviceDefault, hostDefault
#warning "No explicit value set for VIEW_LAYOUTS, using the default of 'deviceDefault, hostDefault'"
#endif

#ifndef DEST_LAYOUTS
#define DEST_LAYOUTS VIEW_LAYOUTS
#endif


namespace LayoutListHelper {
    using left = Kokkos::LayoutLeft;
    using right = Kokkos::LayoutRight;
    using stride = Kokkos::LayoutStride;
    using deviceDefault = Kokkos::DefaultExecutionSpace::array_layout;
    using hostDefault = Kokkos::DefaultHostExecutionSpace::array_layout;

    using LayoutList = decltype(remove_duplicates(TList<VIEW_LAYOUTS>{}));
    using DestLayoutList = decltype(remove_duplicates(TList<DEST_LAYOUTS>{}));
}

using LayoutListHelper::LayoutList;
using LayoutListHelper::DestLayoutList;


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
