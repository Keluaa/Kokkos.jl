
#ifndef KOKKOS_WRAPPER_VIEWS_H
#define KOKKOS_WRAPPER_VIEWS_H

#include "kokkos_wrapper.h"


#ifndef VIEW_DIMENSIONS
/**
 * Controls which `Kokkos::View` dimensions are instantiated.
 * Dimensions are specified as comma separated list of integers.
 *
 * Each dimension adds one pointer for all data types of the views: `Kokkos::View<T*>` in 1D, `Kokkos::View<T**>` in
 * 2D, etc, as well as one more index argument for the `()` operator.
 *
 * The registered method `compiled_dims` returns a tuple of all compiled dimensions.
 */
#define VIEW_DIMENSIONS 1, 2
#endif


#ifndef VIEW_TYPES
/**
 * Controls which `Kokkos::View` types are instantiated.
 * Types are specified as comma separated list of type names.
 *
 * One `Kokkos::View` will be instantiated for each combination of type, dimensions, and memory spaces.
 *
 * The registered method `compiled_types` returns a tuple of all compiled types.
 */
#define VIEW_TYPES double, float, int64_t
#endif


using Idx = typename Kokkos::RangePolicy<>::index_type;

jl_datatype_t* get_idx_type();

void define_kokkos_views(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_VIEWS_H
