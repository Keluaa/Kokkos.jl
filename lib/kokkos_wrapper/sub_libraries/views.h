
#ifndef KOKKOS_WRAPPER_VIEWS_H
#define KOKKOS_WRAPPER_VIEWS_H

#include "kokkos_wrapper.h"
#include "layouts.h"
#include "execution_spaces.h"
#include "parameters.h"


using Dimension = std::integral_constant<int, VIEW_DIMENSION>;


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
    using with_layout = ViewWrap<T, DimCst, OtherLayout, MemSpace, MemTraits>;

    template<typename OtherMemSpace>
    using with_mem_space = ViewWrap<T, DimCst, Layout, OtherMemSpace, MemTraits>;

    static constexpr size_t dim = DimCst::value;

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

#endif //KOKKOS_WRAPPER_VIEWS_H
