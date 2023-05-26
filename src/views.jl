module Views

using CxxWrap
import ..Kokkos: ExecutionSpace, MemorySpace, HostSpace
import ..Kokkos: COMPILED_MEM_SPACES, DEFAULT_DEVICE_MEM_SPACE, DEFAULT_HOST_MEM_SPACE
import ..Kokkos: ensure_kokkos_wrapper_loaded, get_impl_module
import ..Kokkos: memory_space, execution_space, accessible, array_layout, main_space_type, finalize

export Layout, LayoutLeft, LayoutRight, LayoutStride, View, Idx
export COMPILED_TYPES, COMPILED_DIMS, COMPILED_LAYOUTS
export impl_view_type, main_view_type, label, view_wrap, view_data, memory_span, span_is_contiguous
export subview, deep_copy, create_mirror, create_mirror_view


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


"""
    View{T, D, Layout, MemSpace} <: AbstractArray{T, D}

Wrapper around a `Kokkos::View` of `D` dimensions of type `T`, stored in `MemSpace` using the
`Layout`.

Behaves like a normal `Array`. Indexing is done by calling the `Kokkos::View::operator()` function
of the view, and is therefore not highly performant. The best performance with Kokkos views is
achieved by calling Kokkos kernels compiled from C++.

It is supposed that all view accesses are done from the default host execution space. Since the view
may be stored in a memory space different from the host, it may be invalid to access its elements:
if [`accessible`](@ref)`(MemSpace)` is `false`, then all view accesses will throw an error.

Views are created through [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl), which adds
automatically a [finalizer](https://docs.julialang.org/en/v1/base/base/#Base.finalizer) to all
objects which calls the view's destructor when the Julia object is deleted by the garbage collector.

It is important to understand that for a view to be properly disposed of, there is two requirements:
 - the library containing its destructor is still loaded. Views created by the `Kokkos.jl` library
   will always meet this requirement.
 - [`finalize`](@ref) wasn't called.
"""
abstract type View{T, D, L <: Layout, M <: MemorySpace} <: Base.AbstractArray{T, D} end


# Internal functions, defined for each view in 'views.cpp', in 'register_view_types'
function get_ptr end
function get_dims end
function get_strides end


"""
    Idx::Type{<:Integer}

Integer type used by views for indexing on the default execution device. Usually either `Cint` or
`Clonglong`.

Equivalent to `Kokkos::RangePolicy<>::index_type`.

`nothing` if Kokkos is not yet loaded.
"""
Idx = nothing


"""
    COMPILED_DIMS::Tuple{Vararg{Int32}}

List of all View dimensions which are compiled.

By default, dimensions 1 and 2 are compiled.

`nothing` if Kokkos is not yet loaded.
"""
COMPILED_DIMS = nothing


"""
    COMPILED_TYPES::Tuple{Vararg{DataType}}

List of all View element types which are compiled

By default, the following types are compiled: Float64 (double), and Int64 (int64_t).

`nothing` if Kokkos is not yet loaded.
"""
COMPILED_TYPES = nothing


"""
    COMPILED_LAYOUTS::Tuple{Vararg{DataType}}

List of all [`Layouts`](@ref Layout) types which are compiled.

By default, the default array layout for the device and host execution spaces are compiled.

`nothing` if Kokkos is not yet loaded.
"""
COMPILED_LAYOUTS = nothing


function error_view_not_compiled(::Type{View{T, D, L, S}}) where {T, D, L, S}
    ensure_kokkos_wrapper_loaded()

    if !(T in COMPILED_TYPES)
        types_str = join(COMPILED_TYPES, ", ", " and ")
        pluralized_str = length(COMPILED_TYPES) > 1 ? "are" : "is"
        error("view type `$T` is not compiled. Only $types_str $pluralized_str compiled.")
    end

    if !(D in COMPILED_DIMS)
        dims_str = join(string.(COMPILED_DIMS) .* 'D', ", ", " and ")
        pluralized_str = length(COMPILED_DIMS) > 1 ? "are" : "is"
        error("`Kokkos.View$(D)D` cannot be created, as this dimension was not compiled. \
                Only $dims_str $pluralized_str compiled.")
    end

    if !(L in COMPILED_LAYOUTS)
        layout_str = join(COMPILED_LAYOUTS, ", ", " and ")
        pluralized_str = length(COMPILED_LAYOUTS) > 1 ? "are" : "is"
        error("view layout `$L` is not compiled. Only $layout_str $pluralized_str compiled.") 
    end

    if !(S in COMPILED_MEM_SPACES)
        spaces_str = join(COMPILED_MEM_SPACES, ", ", " and ")
        pluralized_str = length(COMPILED_MEM_SPACES) > 1 ? "spaces are" : "space is"
        error("memory space `$S` is not compiled. The only compiled $pluralized_str $spaces_str.")
    end

    error("$(D)D views of type $T stored in $S with a $L are not compiled.")
end


# Instances of `alloc_view` for each compiled type, dimension, layout and memory space are defined
# in 'views.cpp', in 'register_constructor'.
function alloc_view(::Type{View{T, D, L, S}},
        dims::Dims{D}, mem_space, layout, label, zero_fill, dim_pad) where {T, D, L, S}
    # Fallback: error handler
    error_view_not_compiled(View{T, D, L, S})
end


"""
    impl_view_type(::Type{View{T, D, L, S}})

Returns the internal [`View`](@ref) type for the given complete View type.

The opposite of [`main_view_type`](@ref).

```julia-repl
julia> view_t = Kokkos.View{Float64, 2, Kokkos.LayoutRight, Kokkos.HostSpace}
Kokkos.Views.View{Float64, 2, Kokkos.Views.LayoutRight, Kokkos.Spaces.HostSpace}

julia> view_impl_t = Kokkos.impl_view_type(view_t)
Kokkos.KokkosWrapper.Impl.View2D_R_HostAllocated{Float64}

julia> supertype(supertype(view_impl_t))
Kokkos.Views.View{Float64, 2, Kokkos.Views.LayoutRight, Kokkos.KokkosWrapper.Impl.HostSpaceImplAllocated}

julia> view_impl_t <: view_t  # Julia types are contra-variant
false
```
"""
function impl_view_type(::Type{View{T, D, L, S}}) where {T, D, L, S}
    # Fallback: error handler
    error_view_not_compiled(View{T, D, L, S})
end


"""
    main_view_type(::View)
    main_view_type(::Type{<:View})

The "main type" of the view: converts `Type{View1D_S_HostAllocated{Float64}}` into
`Type{View{Float64, 1, LayoutStride, HostSpace}}`, which is easier to understand.

The opposite of [`impl_view_type`](@ref).
"""
main_view_type(::Type{<:View{T, D, L, S}}) where {T, D, L, S} = View{T, D, L, main_space_type(S)}
main_view_type(v::View) = main_view_type(supertype(supertype(typeof(v))))


"""
    accessible(::View)

Return `true` if the view is accessible from the default host execution space.
"""
accessible(::View{T, D, L, MemSpace}) where {T, D, L, MemSpace} = accessible(MemSpace)


"""
    array_layout(::View)

Return the [`Layout`](@ref) type of the view.
"""
array_layout(::View{T, D, L, M}) where {T, D, L, M} = L


"""
    memory_space(::View)

The memory space type in which the view data is stored.

```julia-repl
julia> my_cuda_space = Kokkos.CudaSpace()
 ...

julia> v = View{Float64}(undef, 10; mem_space=my_cuda_space)
 ...

julia> memory_space(v)
Kokkos.Spaces.CudaSpace
```
"""
memory_space(::View{T, D, L, MemSpace}) where {T, D, L, MemSpace} = main_space_type(MemSpace)


@generated function jit_deep_copy(
    dst::View{T, D, Dst_L, Dst_S},
    src::View{T, D, Src_L, Src_S}
) where {T, D, Dst_L, Dst_S, Src_L, Src_S}
    func_sym = Kokkos.KokkosWrapper.get_symbol_for_prototype(
        :deep_copy,
        Cvoid,
        (View{T, D, Dst_L, Dst_S}, View{T, D, Src_L, Src_S})
    )

    return quote
        func_ptr = Kokkos.KokkosWrapper.get_function_ptr($(func_sym))

        if func_ptr === nothing
            func_ptr = Kokkos.KokkosWrapper.compile_and_load_function(
                $(func_sym);
                view_types = (T,), view_dims = (D,), view_layouts = (Src_L, Dst_L),
                mem_spaces = (Dst_S, Src_S), exec_spaces = ()
            )
        end

        ccall(func_ptr,
            Cvoid, (Ref{View{T, D, Dst_L, Dst_S}}, Ref{View{T, D, Src_L, Src_S}}),
            dst, src
        )
    end
end


"""
    label(::View)

Return the label of the `View`.
"""
function label end


"""
    view_data(::View)

The pointer to the data of the `View`. Using `Base.pointer(view)` is preferred over this method.

Equivalent to `view.data()`.
"""
function view_data end


"""
    memory_span(::View)

Total size of the view data in memory, in bytes.

Equivalent to `view.impl_map().memory_span()`.
"""
function memory_span end


"""
    span_is_contiguous(::View)

`true` if the view stores all its elements contiguously in memory.

Equivalent to [`view.span_is_contiguous()`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/view.html?highlight=span_is_contiguous#_CPPv4NK18span_is_contiguousEv).
"""
function span_is_contiguous end


"""
    deep_copy(dest::View, src::View)
    deep_copy(space::ExecutionSpace, dest::View, src::View)

Performs a copy of all data from `src` to `dest`.

In order for the copy to be possible, both views must have the same dimension, and either have the
same layout or are both accessible from `space`.

If a `space` is given, the copy may be asynchronous. If not the copy will be synchronous.

Equivalent to `Kokkos::deep_copy(dest, src)` or `Kokkos::deep_copy(space, dest, src)`.
[See the Kokkos docs about `Kokkos::deep_copy`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/deep_copy.html#deep-copy)
"""
function deep_copy end


"""
    create_mirror(src::View; mem_space = nothing, zero_fill = false)

Create a new [`View`](@ref) in the same way as `similar(src)`, with the same layout and padding as
`src`.

If `mem_space` is `nothing` the new view will be in a memory space accessible by the host, otherwise
it must be a memory space instance where the new view will be allocated.

If `zero_fill` is true, the new view will have all of its elements set to their default value.

[See the Kokkos docs about `Kokkos::create_mirror`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/create_mirror.html)
"""
function create_mirror(src::View; mem_space = nothing, zero_fill = false)
    return create_mirror(src, mem_space, zero_fill)
end


"""
    create_mirror_view(src::View; mem_space = nothing, zero_fill = false)

Equivalent to [`create_mirror`](@ref), but if `src` is already accessible by the host, `src` is
returned and no view is created.
"""
function create_mirror_view(src::View; mem_space = nothing, zero_fill = false)
    return create_mirror_view(src, mem_space, zero_fill)
end


"""
    subview(v::View, indexes...)
    subview(v::View, indexes::Tuple{Vararg{Union{Int, Colon, AbstractUnitRange}}})

Return a new `Kokkos.view` which will be a subview into the region specified by `indexes` of `v`,
with the same memory space (but maybe not the same layout).

Unspecified dimensions are completed by `:`, e.g. if `v` is a 3D view `(1,)` and `(1, :, :)` will
return the same subview.

A subview may need `LayoutStride` to be compiled in order to be represented.

Equivalent to [`Kokkos::subview`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/subview.html).

`Kokkos::ALL` is equivalent to `:`.

# Example
```julia-repl
julia> v = Kokkos.View{Float64}(undef, 4, 4);

julia> v[:] .= collect(1:length(v));

julia> v
4×4 Kokkos.KokkosWrapper.Impl.View2D_R_HostAllocated{Float64}:
 1.0  5.0   9.0  13.0
 2.0  6.0  10.0  14.0
 3.0  7.0  11.0  15.0
 4.0  8.0  12.0  16.0

julia> Kokkos.subview(v, (2:3, 2:3))
2×2 Kokkos.KokkosWrapper.Impl.View2D_R_HostAllocated{Float64}:
 6.0  10.0
 7.0  11.0

julia> Kokkos.subview(v, (:, 1))  # The subview may change its layout to `LayoutStride` 
4-element Kokkos.KokkosWrapper.Impl.View1D_S_HostAllocated{Float64}:
 1.0
 2.0
 3.0
 4.0

julia> Kokkos.subview(v, (1,))  # Equivalent to `(1, :)`
4-element Kokkos.KokkosWrapper.Impl.View1D_R_HostAllocated{Float64}:
  1.0
  5.0
  9.0
 13.0
```

# !!! warning

    `Kokkos.subview` is __not__ equivalent to `Base.view`, as it returns a new `Kokkos.View` object,
    while `Base.view` returns a `SubArray`, which cannot be passed to a `ccall`.

"""
function subview(::View{T, D, L, S}, ::Tuple, subview_dim::Type{Val{SD}}, subview_layout::Type{SL}) where {T, D, L, S, SD, SL}
    # Fallback: error handler
    error_view_not_compiled(View{T, SD, SL, S})
end


function _get_subview_dim_and_layout(src_rank, src_layout, indexes_type)
    # IMPORTANT: this should always return the same layout type as Kokkos::Subview.
    # The code here is equivalent to the logic of Kokkos::Impl::ViewMapping at
    # https://github.com/kokkos/kokkos/blob/62d2b6c879b74b6ae7bd06eb3e5e80139c4708e6/core/src/impl/Kokkos_ViewMapping.hpp#L3812-L3885

    src_rank == 0 && return 0, src_layout

    is_range = (!==).(indexes_type.parameters, Int)
    if length(is_range) < src_rank
        # Select the full range of all remaining dimensions
        append!(is_range, Iterators.repeated(true, src_rank - length(is_range)))
    end

    rank = sum(is_range)
    rank == 0 && return 0, src_layout

    keep_layout = rank <= 2 &&
        ((src_layout === LayoutLeft  && is_range[1]       ) ||
         (src_layout === LayoutRight && is_range[src_rank]))

    return rank, (keep_layout ? src_layout : LayoutStride)
end


function subview(v::View{T, D, L, S}, indexes::Tuple{Vararg{Union{Int, Colon, AbstractUnitRange}}}) where {T, D, L, S}
    subview_dim, subview_layout = _get_subview_dim_and_layout(D, L, typeof(indexes))
    return subview(v, indexes, Val{subview_dim}, subview_layout)
end


subview(v::View, indexes::Vararg{Union{Int, Colon, AbstractUnitRange}}) = subview(v, indexes)


# === Constructors ===

function _get_mem_space_type(mem_space)
    mem_space_t::DataType = Nothing
    if mem_space isa DataType
        # Default construction of the memory space (delegated to `alloc_view`)
        if mem_space <: ExecutionSpace
            mem_space_t = memory_space(mem_space)
        else
            mem_space_t = mem_space
        end
    elseif mem_space isa MemorySpace
        mem_space_t = main_space_type(typeof(mem_space))
    else
        ensure_kokkos_wrapper_loaded()
        throw(TypeError(:View, "constructor", Union{DataType, MemorySpace}, mem_space))
    end
    return mem_space_t
end


function _get_layout_type(layout, mem_space_t)
    layout_t::DataType = Nothing
    if layout isa DataType
        layout_t = layout
    elseif layout isa Layout
        layout_t = typeof(layout)
    elseif layout === nothing
        layout_t = array_layout(execution_space(mem_space_t))
    else
        ensure_kokkos_wrapper_loaded()
        throw(TypeError(:View, "constructor", Union{DataType, Layout}, layout))
    end
    return layout_t
end


"""
    View{T, D, Layout, MemSpace}(dims;
        mem_space = DEFAULT_DEVICE_MEM_SPACE,
        layout = nothing,
        label = "",
        zero_fill = true,
        dim_pad = false
    )

Construct an N-dimensional `View{T}`.

`D` can be deduced from `dims`, which can either be a `NTuple{D, Integer}` or `D` integers.

`Layout` defaults to the type of `layout`. `layout` can be a [`Layout`](@ref) instance or type.
If `layout` is `nothing` it defaults to [`array_layout(execution_space(mem_space))`](@ref array_layout)
after `mem_space` is converted to a `MemorySpace`.

`MemSpace` defaults to the type of `mem_space`. `mem_space` can be an [`ExecutionSpace`](@ref), in
which case it is converted to a [`MemorySpace`](@ref) with [`memory_space`](@ref). `mem_space` can
be either an instance of a [`MemorySpace`](@ref), or one of the main types of memory spaces, in
which case an instance of a [`MemorySpace`](@ref) is default constructed (behaviour of Kokkos by
default).

The `label` is the debug label of the view.

If `zero_fill=true`, all elements will be set to `0`. Uses `Kokkos::WithoutInitializing` internally.

`dim_pad` controls the padding of dimensions. Uses `Kokkos::AllowPadding` internally. If `true`,
then a view might not have a layout identical to a classic `Array`, for better memory alignment.

See [the Kokkos documentation about `Kokkos::view_alloc()`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/view_alloc.html)
for more info.
"""
function View{T, D, L, S}(dims::Dims{D};
    mem_space = DEFAULT_DEVICE_MEM_SPACE,
    layout = nothing,
    label = "",
    zero_fill = true,
    dim_pad = false
) where {T, D, L, S}
    mem_space_t = mem_space isa DataType ? mem_space : typeof(mem_space)
    if !(main_space_type(mem_space_t) <: S)
        error("Conficting types for `View` constructor typing `$S` and `mem_space` kwarg: $mem_space \
               (type: $(main_space_type(mem_space_t)))")
    end

    if isnothing(layout)
        # ok
    elseif layout isa DataType
        layout !== L && error("expected `layout` to be a $L type or an instance, got: $layout")
    elseif !(layout isa L)
        error("Conficting types for `View` constructor typing `$L` and `layout` kwargs: $layout \
               (type: $(typeof(layout)))")
    end

    if L === LayoutStride && !(layout isa LayoutStride)
        error("the `View` constructor with a `LayoutStride` requires a instance of the layout")
    end

    if mem_space isa DataType
        # Let `alloc_view` call the memory space constructor
        return alloc_view(View{T, D, L, S}, dims, nothing, layout, label, zero_fill, dim_pad)
    else
        return alloc_view(View{T, D, L, S}, dims, mem_space, layout, label, zero_fill, dim_pad)
    end
end

View{T, D, L, S}(::UndefInitializer, dims::Dims{D}; kwargs...) where {T, D, L, S} =
    View{T, D, L, S}(dims; kwargs..., zero_fill=false)


# View{T, D, L} to View{T, D, L, S}
function View{T, D, L}(dims::Dims{D};
    mem_space = DEFAULT_DEVICE_MEM_SPACE,
    kwargs...
) where {T, D, L}
    mem_space_t = _get_mem_space_type(mem_space)
    return View{T, D, L, mem_space_t}(dims; mem_space, kwargs...)
end

View{T, D, L}(::UndefInitializer, dims::Dims{D}; kwargs...) where {T, D, L} =
    View{T, D, L}(dims; kwargs..., zero_fill=false)


# View{T, D} to View{T, D, L, S}
function View{T, D}(dims::Dims{D};
    mem_space = DEFAULT_DEVICE_MEM_SPACE,
    layout = nothing,
    kwargs...
) where {T, D}
    mem_space_t = _get_mem_space_type(mem_space)
    layout_t = _get_layout_type(layout, mem_space_t)
    return View{T, D, layout_t, mem_space_t}(dims; mem_space, layout, kwargs...)
end

View{T, D}(::UndefInitializer, dims::Dims{D}; kwargs...) where {T, D} =
    View{T, D}(dims; kwargs..., zero_fill=false)


# Int... to Dims{D}
"""
    View{T, D, L, S}(undef, dims; kwargs...)
    View{T, D, L}(undef, dims; kwargs...)
    View{T, D}(undef, dims; kwargs...)
    View{T}(undef, dims; kwargs...)

Construct an N-dimensional `View{T}`, without initialization of its elements.

Strictly equivalent to passing `zero_fill=false` to the `kwargs`.
"""
View{T, D, L, S}(::UndefInitializer, dims::Integer...; kwargs...) where {T, D, L, S} =
    View{T, D, L, S}(convert(Tuple{Vararg{Int}}, dims); kwargs..., zero_fill=false)
View{T, D, L}(::UndefInitializer, dims::Integer...; kwargs...) where {T, D, L} =
    View{T, D, L}(convert(Tuple{Vararg{Int}}, dims); kwargs..., zero_fill=false)
View{T, D}(::UndefInitializer, dims::Integer...; kwargs...) where {T, D} =
    View{T, D}(convert(Tuple{Vararg{Int}}, dims); kwargs..., zero_fill=false)

# Int... to Dims{D} but without the UndefInitializer
View{T, D, L, S}(dims::Integer...; kwargs...) where {T, D, L, S} =
    View{T, D, L, S}(convert(Tuple{Vararg{Int}}, dims); kwargs...)
View{T, D, L}(dims::Integer...; kwargs...) where {T, D, L} =
    View{T, D, L}(convert(Tuple{Vararg{Int}}, dims); kwargs...)
View{T, D}(dims::Integer...; kwargs...) where {T, D} =
    View{T, D}(convert(Tuple{Vararg{Int}}, dims); kwargs...)

# Deduce `D` from the number of Ints
View{T}(::UndefInitializer, dims::Dims{D}; kwargs...) where {T, D} =
    View{T, D}(dims; kwargs..., zero_fill=false)
View{T}(::UndefInitializer, dims::Integer...; kwargs...) where {T} =
    View{T}(convert(Tuple{Vararg{Int}}, dims); kwargs..., zero_fill=false)
View{T}(dims::Dims{D}; kwargs...) where {T, D} =
    View{T, D}(dims; kwargs...)
View{T}(dims::Integer...; kwargs...) where {T} =
    View{T}(convert(Tuple{Vararg{Int}}, dims); kwargs...)

# Empty constructors
View{T, D, L, S}(; kwargs...) where {T, D, L, S} =
    View{T, D, L, S}(ntuple(Returns(0), D); kwargs..., zero_fill=false)
View{T, D, L}(; kwargs...) where {T, D, L} =
    View{T, D, L}(ntuple(Returns(0), D); kwargs..., zero_fill=false)
View{T, D}(; kwargs...) where {T, D} =
    View{T, D}(ntuple(Returns(0), D); kwargs..., zero_fill=false)
View{T}(; kwargs...) where {T} =
    View{T}((0,); kwargs..., zero_fill=false)


# Instances of `view_wrap` for each compiled type, dimension and memory space are defined in
# 'views.cpp', in 'register_constructor'.
"""
    view_wrap(array::DenseArray{T, D})
    view_wrap(::Type{View{T, D}}, array::DenseArray{T, D})

    view_wrap(array::AbstractArray{T, D})
    view_wrap(::Type{View{T, D}}, array::AbstractArray{T, D})

    view_wrap(::Type{View{T, D, L, S}}, d::NTuple{D, Int}, p::Ptr{T}; layout = nothing)

Construct a new [`View`](@ref) from the data of a Julia-allocated array (or from any valid array or
pointer).
The returned view does not own the data: no copy is made.

The memory space `S` is `HostSpace` when `array` is a `DenseArray` or `AbstractArray`, and the
layout `L` is [`LayoutLeft`](@ref) for `DenseArray` and [`LayoutStride`](@ref) for `AbstractArray`.

If `L` is `LayoutStride`, then the kwarg `layout` should be an instance of a `LayoutStride` which
specifies the stride of each dimension.

!!! warning

    Julia arrays have a column-major layout by default. This correspond to a [`LayoutLeft`](@ref),
    while Kokkos prefers [`LayoutRight`](@ref) for CPU allocated arrays.
    If `strides(array) ≠ strides(view_wrap(array))` then it might lead to segfaults.
    This only concerns 2D arrays and above.

!!! warning

    The returned view does not hold a reference to the original array.
    It is the responsability of the user to make sure the original array is kept alive as long as
    the view should be accessed.
"""
view_wrap(::Type{View{T, D, L, S}}, ::Dims{D}, layout, ::Ptr{T}) where {T, D, L, S} =
    error_view_not_compiled(View{T, D, L, S})
view_wrap(::Type{View{T, D, L, S}}, d::Dims{D}, p::Ptr{T}; layout=nothing) where {T, D, L, S} =
    view_wrap(View{T, D, L, S}, d, layout, p)

view_wrap(::Type{View{T, D, L, S}}, a::AbstractArray{T, D}; kwargs...) where {T, D, L, S} =
    view_wrap(View{T, D, L, S}, size(a), pointer(a); kwargs...)

# From a Julia-allocated DenseArray (column-major in host space)
view_wrap(::Type{View{T, D}}, a::DenseArray{T, D}) where {T, D} =
    view_wrap(View{T, D, LayoutLeft, HostSpace}, a)
view_wrap(a::DenseArray{T, D}) where {T, D} =
    view_wrap(View{T, D}, a)

# From any AbstractArray (LayoutStride in host space)
view_wrap(::Type{View{T, D}}, a::AbstractArray{T, D}) where {T, D} =
    view_wrap(View{T, D, LayoutStride, HostSpace}, a; layout=LayoutStride(strides(a)))
view_wrap(a::AbstractArray{T, D}) where {T, D} =
    view_wrap(View{T, D}, a)


# === Array interface ===

Base.IndexStyle(::Type{<:View}) = IndexCartesian()

@inline Base.size(v::View) = get_dims(v)

@inline to_c_index(I::Vararg{Int, D}) where {D} = convert.(Idx, I .- 1)
Base.@propagate_inbounds elem_ptr(v::View{T, D}, I::Vararg{Int, D}) where {T, D} =
    (@boundscheck checkbounds(v, I...); get_ptr(v, to_c_index(I...)...))

Base.@propagate_inbounds Base.getindex(v::View{T, D}, I::Vararg{Int, D}) where {T, D} =
    elem_ptr(v, I...)[]

Base.@propagate_inbounds Base.setindex!(v::View{T, D}, val, I::Vararg{Int, D}) where {T, D} =
    (elem_ptr(v, I...)[] = convert(T, val))

Base.similar(a::View{T, D, L, M}) where {T, D, L, M} =
    View{T, D, L, main_space_type(M)}(size(a);
        zero_fill=false, layout=L <: LayoutStride ? LayoutStride(strides(a)) : nothing)
Base.similar(a::View{T, D, L, M}, dims::Dims{D}) where {T, D, L, M} =
    View{T, D, L, main_space_type(M)}(dims;
        zero_fill=false, layout=L <: LayoutStride ? LayoutStride(strides(a)) : nothing)
Base.similar(::View{T, <:Any, L, M}, dims::Dims{D}) where {T, D, L, M} =
    View{T, D, L, main_space_type(M)}(dims;
        zero_fill=false, layout=L <: LayoutStride ? LayoutStride(Base.size_to_strides(1, dims...)) : nothing)
Base.similar(::View{<:Any, <:Any, L, M}, ::Type{T}, dims::Dims{D}) where {T, D, L, M} =
    View{T, D, L, main_space_type(M)}(dims;
        zero_fill=false, layout=L <: LayoutStride ? LayoutStride(Base.size_to_strides(1, dims...)) : nothing)


Base.copyto!(dest::View{DT, Dim, DL, DM}, src::View{ST, Dim, SL, SM}) where {DT, ST, DL, SL, DM, SM, Dim} =
    deep_copy(dest, src)


Base.sizeof(v::View) = Int(memory_span(v))


# === Pointer conversion ===

# Pointer to the array data
Base.pointer(v::V) where {T, V <: View{T}} = Ptr{T}(view_data(v).cpp_object)


# Pointer to the view object, for ccalls:

# For the case `my_func(v::V) where V = ccall(my_c_func, (Ref{V},), v)`
Base.cconvert(::Type{Ref{V}}, v::V) where {V <: View} = Ptr{Nothing}(v.cpp_object)

# For the case `my_func(v) = ccall(my_c_func, (Ref{View{...}},), v)`
function Base.cconvert(::Type{Ref{View{T, D, L, S}}}, v::View) where {T, D, L, S}
    impl_view_t = impl_view_type(View{T, D, L, S})
    if v isa impl_view_t
        return Ptr{Nothing}(v.cpp_object)
    else
        error("Expected a view of type `$impl_view_t` (aka `View{$T, $D, $L, $S}`), got: `$(typeof(v))`")
    end
end


# === Strided Array interface ===

Base.strides(v::View) = get_strides(v)

Base.unsafe_convert(::Type{Ptr{T}}, v::V) where {T, V <: View{T}} = pointer(v)

Base.elsize(::Type{<:View{T}}) where {T} = sizeof(T)


function __init_vars()
    impl = get_impl_module()
    global Idx = Base.invokelatest(impl.__idx_type)
    global COMPILED_DIMS = Base.invokelatest(impl.__compiled_dims)
    global COMPILED_TYPES = Base.invokelatest(impl.__compiled_types)
    global COMPILED_LAYOUTS = Base.invokelatest(impl.__compiled_layouts)
end

end