
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

using DimensionsToInstantiate = std::integer_sequence<int, VIEW_DIMENSIONS>;


/**
 * Sets `type` to `T` followed by `N` pointers: `add_pointers<int, 2>::type == int**`
 */
template<typename T, int N>
struct add_pointers { using type = typename add_pointers<std::add_pointer_t<T>, N-1>::type; };

template<typename T>
struct add_pointers<T, 0> { using type = T; };


/**
 * Basic wrapper around a `Kokkos::View`, mostly providing convenience functionalities over dimensions and the data type
 * of the view.
 *
 * It is this type that is registered with CxxWrap, not Kokkos::View, therefore all Julia methods defined with CxxWrap
 * should use this type in their arguments / return type, not Kokkos::View.
 */
template<typename T, typename DimCst, typename MemSpace,
         typename T_Ptr = typename add_pointers<T, DimCst::value>::type,
         typename KokkosViewT = typename Kokkos::View<T_Ptr, MemSpace>>
struct ViewWrap : public KokkosViewT
{
    using type = T;
    using mem_space = MemSpace;
    using kokkos_view_t = KokkosViewT;

    static constexpr size_t dim = DimCst::value;

    using IdxTuple [[maybe_unused]] = decltype(std::tuple_cat(std::array<Idx, dim>()));

    // Should be 'using typename KokkosViewT::View;' but compiler incompatibilities make this impossible
    using Kokkos::View<T_Ptr, MemSpace>::View;

    explicit ViewWrap(const KokkosViewT& other) : KokkosViewT(other) {};
    explicit ViewWrap(KokkosViewT&& other) : KokkosViewT(std::move(other)) {};

    [[nodiscard]] std::array<int64_t, dim> get_dims() const {
        std::array<int64_t, dim> dims{};
        for (size_t i = 0; i < dim; i++) {
            dims.at(i) = this->extent_int(i);
        }
        return dims;
    }
};


jl_datatype_t* get_idx_type();

void define_kokkos_views(jlcxx::Module& mod);

#endif //KOKKOS_WRAPPER_VIEWS_H
