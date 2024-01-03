
"""
    Layout

Abstract super-type of all view layouts.

Sub-types:
 - [`LayoutLeft`](@ref)
 - [`LayoutRight`](@ref)
 - [`LayoutStride`](@ref)
"""
abstract type Layout end


"""
    LayoutLeft

Fortran-style column major array layout. This is also the layout of Julia arrays.

While indexing, the first index is the contiguous one: `v[i0, i1, i2]`.

Equivalent to [`Kokkos::LayoutLeft`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/layoutLeft.html).
"""
struct LayoutLeft <: Layout end


"""
    LayoutRight

C-style row major array layout.

While indexing, the last index is the contiguous one: `v[i2, i1, i0]`.

Equivalent to [`Kokkos::LayoutRight`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/layoutRight.html).
"""
struct LayoutRight <: Layout end


"""
    LayoutStride

Arbitrary array layout, mostly used for sub-views.

Equivalent to [`Kokkos::LayoutStride`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/layoutStride.html).

When building a new view with a `LayoutStride`, the strides of each dimension must be given to the
view constructor:
```julia
# A 7×11 matrix with column-major layout
v = Kokkos.View{Float64}(undef, 7, 11; layout=LayoutStride(1, 11))

# A 7×11 matrix with row-major layout
v = Kokkos.View{Float64}(undef, 7, 11; layout=LayoutStride(7, 1))
```

This differs slightly from the C++ Kokkos constructor, where dimensions and strides are interleaved.
"""
struct LayoutStride <: Layout
    strides::Dims
end

LayoutStride() = LayoutStride(())
LayoutStride(strides::Integer...) = LayoutStride(convert(Tuple{Vararg{Int}}, strides))
