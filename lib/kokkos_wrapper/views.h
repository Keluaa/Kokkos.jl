
#ifndef KOKKOS_WRAPPER_VIEWS_H
#define KOKKOS_WRAPPER_VIEWS_H

#include "kokkos_wrapper.h"
#include "layouts.h"
#include "execution_spaces.h"

#ifndef WRAPPER_BUILD
#include "parameters.h"
#endif


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
#warning "No explicit value set for VIEW_DIMENSIONS, using the default of '1, 2'"
#endif


#ifndef VIEW_TYPES
/**
 * Controls which `Kokkos::View` types are instantiated.
 * Types are specified as comma separated list of type names.
 *
 * One `Kokkos::View` will be instantiated for each combination of type, dimensions, layout, and memory spaces.
 *
 * The registered method `compiled_types` returns a tuple of all compiled types.
 */
#define VIEW_TYPES double, int64_t
#warning "No explicit value set for VIEW_TYPES, using the default of 'double, int64_t'"
#endif


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
 *
 * Importantly, the `Kokkos::View` type is complete: it has the same parameters as the type returned by `Kokkos::subview`
 * and can represent any `Kokkos::View` exactly.
 */
template<typename T, typename DimCst, typename LayoutType, typename MemSpace,
         typename MemTraits = Kokkos::MemoryTraits<0>,
         typename T_Ptr = typename add_pointers<T, DimCst::value>::type,
         typename Device = typename MemSpace::device_type,
         typename KokkosViewT = typename Kokkos::View<T_Ptr, LayoutType, Device, MemTraits>>
struct ViewWrap : public KokkosViewT
{
    using type = T;
    using layout = LayoutType;
    using mem_space = MemSpace;

    using kokkos_view_t = KokkosViewT;

    template<typename OtherLayout>
    using with_layout = ViewWrap<T, DimCst, OtherLayout, MemSpace>;  // TODO: replace MemSpace with Device

    static constexpr size_t dim = DimCst::value;

    using IdxTuple = decltype(std::tuple_cat(std::array<Idx, dim>()));

#ifdef __INTEL_COMPILER
    template<typename... Args>
    ViewWrap(Args&&... args) : KokkosViewT(std::forward<Args>(args)...) {}
#else
    // Should be 'using typename KokkosViewT::View;' but compiler incompatibilities make this impossible
    using Kokkos::View<T_Ptr, LayoutType, Device, MemTraits>::View;
#endif // __INTEL_COMPILER

    explicit ViewWrap(const KokkosViewT& other) : KokkosViewT(other) {};
    explicit ViewWrap(KokkosViewT&& other) : KokkosViewT(std::move(other)) {};

    [[nodiscard]] std::array<int64_t, dim> get_dims() const {
        std::array<int64_t, dim> dims{};
        for (size_t i = 0; i < dim; i++) {
            dims.at(i) = this->extent_int(i);
        }
        return dims;
    }

    [[nodiscard]] std::array<int64_t, dim> get_strides() const {
        std::array<int64_t, dim> strides{};
        for (size_t i = 0; i < dim; i++) {
            strides.at(i) = this->stride(i);
        }
        return strides;
    }
};


#if defined(WRAPPER_BUILD) && COMPLETE_BUILD == 1
void define_kokkos_views(jlcxx::Module& mod);
#else
void define_kokkos_views(jlcxx::Module&) {}
#endif

#endif //KOKKOS_WRAPPER_VIEWS_H
