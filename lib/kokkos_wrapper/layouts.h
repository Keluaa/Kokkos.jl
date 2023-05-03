#ifndef KOKKOS_WRAPPER_LAYOUTS_H
#define KOKKOS_WRAPPER_LAYOUTS_H

#include "Kokkos_Core.hpp"
#include "jlcxx/jlcxx.hpp"
#include "utils.h"


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
#endif


namespace LayoutListHelper {
    using left = Kokkos::LayoutLeft;
    using right = Kokkos::LayoutRight;
    using stride = Kokkos::LayoutStride;
    using deviceDefault = Kokkos::DefaultExecutionSpace::array_layout;
    using hostDefault = Kokkos::DefaultHostExecutionSpace::array_layout;

    using LayoutList = decltype(remove_duplicates(TList<VIEW_LAYOUTS>{}));
}

using LayoutListHelper::LayoutList;


void define_all_layouts(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_LAYOUTS_H
