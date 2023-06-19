module Views

using CxxWrap
import ..Kokkos: DynamicCompilation
import ..Kokkos: ExecutionSpace, MemorySpace, HostSpace
import ..Kokkos: ENABLED_MEM_SPACES, DEFAULT_DEVICE_MEM_SPACE, DEFAULT_HOST_MEM_SPACE, Idx
import ..Kokkos: ensure_kokkos_wrapper_loaded, get_impl_module
import ..Kokkos: memory_space, execution_space, accessible, array_layout, main_space_type, finalize

export Layout, LayoutLeft, LayoutRight, LayoutStride, View, Idx
export impl_view_type, main_view_type, label, view_wrap, view_data, memory_span, span_is_contiguous
export subview, deep_copy, create_mirror, create_mirror_view


# TODO: move Layouts to the main module
# TODO: always pass 'right, left, stride' when compiling the wrapper

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


function _extract_view_params(view_t::Type{<:View})
    return [eltype(view_t)], [ndims(view_t)], [array_layout(view_t)], [memory_space(view_t)]
end


const _COMPILED_VIEW_TYPES = Set{Type}()


function compile_view(view_t::Type{<:View}; for_function=nothing, no_error=false)
    view_t = main_view_type(view_t)

    if view_t in _COMPILED_VIEW_TYPES
        # This view type should already be compiled
        if no_error
            return false
        elseif !isnothing(for_function)
            error("View type `$view_t` is already compiled, but `$for_function` has no specific \
                   method for it")
        else
            error("View type `$view_t` is already compiled, but there is no specific method for it")
        end
    end

    view_types, view_dims, view_layouts, mem_spaces = _extract_view_params(view_t)
    DynamicCompilation.compile_and_load(@__MODULE__, "views";
        view_types, view_dims, view_layouts, mem_spaces
    )

    push!(_COMPILED_VIEW_TYPES, view_t)

    return true
end


function alloc_view(
    view_t::Type{<:View},
    dims::Dims, mem_space, layout,
    label, zero_fill, dim_pad
)
    @nospecialize view_t dims mem_space layout label zero_fill dim_pad
    return DynamicCompilation.@compile_and_call(
        alloc_view, (view_t, dims, mem_space, layout, label, zero_fill, dim_pad),
        begin
            if ndims(view_t) != length(dims)
                error("Dimension mismatch: view type is $(ndims(view_t))D ($view_t), \
                       got $(length(dims))D dims tuple: $dims")
            end
            compile_view(view_t; for_function=alloc_view, no_error=true)
        end
    )
end


function _get_ptr(@nospecialize(v::View), @nospecialize(i::Vararg{Integer}))
    return DynamicCompilation.@compile_and_call(_get_ptr, (v, i...),
        compile_view(typeof(v); for_function=_get_ptr, no_error=true)
    )
end

_get_ptr(v::View, i::Tuple{Vararg{Integer}}) = _get_ptr(v, i...)


function _get_dims(@nospecialize(v::View))
    return DynamicCompilation.@compile_and_call(_get_dims, (v,),
        compile_view(typeof(v); for_function=_get_dims, no_error=true)
    )
end


function _get_strides(@nospecialize(v::View))
    return DynamicCompilation.@compile_and_call(_get_strides, (v,),
        compile_view(typeof(v); for_function=_get_strides, no_error=true)
    )
end


function get_tracker(@nospecialize(v::View))
    return DynamicCompilation.@compile_and_call(get_tracker, (v,),
        compile_view(typeof(v); for_function=get_tracker, no_error=true)
    )
end


"""
    impl_view_type(::Type{View{T, D, L, S}})

Returns the internal [`View`](@ref) type for the given complete View type.

The opposite of [`main_view_type`](@ref).

```julia-repl
julia> view_t = Kokkos.View{Float64, 2, Kokkos.LayoutRight, Kokkos.HostSpace}
Kokkos.Views.View{Float64, 2, Kokkos.Views.LayoutRight, Kokkos.Spaces.HostSpace}

julia> view_impl_t = Kokkos.impl_view_type(view_t)
Kokkos.Wrapper.Impl.View2D_R_HostAllocated{Float64}

julia> supertype(supertype(view_impl_t))
Kokkos.Views.View{Float64, 2, Kokkos.Views.LayoutRight, Kokkos.Wrapper.Impl.HostSpaceImplAllocated}

julia> view_impl_t <: view_t  # Julia types are contra-variant
false
```
"""
function impl_view_type(@nospecialize(view_t::Type{<:View}))
    return DynamicCompilation.@compile_and_call(impl_view_type, (view_t,),
        compile_view(view_t; for_function=impl_view_type, no_error=true)
    )
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
    accessible(::Type{<:View})

Return `true` if the view is accessible from the default host execution space.
"""
accessible(v::View) = accessible(typeof(v))
accessible(::Type{<:View{T, D, L, MemSpace}}) where {T, D, L, MemSpace} = accessible(MemSpace)


"""
    array_layout(::View)
    array_layout(::Type{<:View})

Return the [`Layout`](@ref) type of the view.
"""
array_layout(v::View) = array_layout(typeof(v))
array_layout(::Type{<:View{T, D, L, M}}) where {T, D, L, M} = L



"""
    memory_space(::View)
    memory_space(::Type{<:View})

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
memory_space(v::View) = memory_space(typeof(v))
memory_space(::Type{<:View{T, D, L, MemSpace}}) where {T, D, L, MemSpace} = main_space_type(MemSpace)


"""
    label(::View)

Return the label of the `View`.
"""
function label(@nospecialize(v::View))
    return DynamicCompilation.@compile_and_call(label, (v,),
        compile_view(typeof(v); for_function=label, no_error=true)
    )
end


"""
    view_data(::View)

The pointer to the data of the `View`. Using `Base.pointer(view)` is preferred over this method.

Equivalent to `view.data()`.
"""
function view_data(@nospecialize(v::View))
    return DynamicCompilation.@compile_and_call(view_data, (v,),
        compile_view(typeof(v); for_function=view_data, no_error=true)
    )
end


"""
    memory_span(::View)

Total size of the view data in memory, in bytes.

Equivalent to `view.impl_map().memory_span()`.
"""
function memory_span(@nospecialize(v::View))
    return DynamicCompilation.@compile_and_call(memory_span, (v,),
        compile_view(typeof(v); for_function=memory_span, no_error=true)
    )
end


"""
    span_is_contiguous(::View)

`true` if the view stores all its elements contiguously in memory.

Equivalent to [`view.span_is_contiguous()`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/view.html?highlight=span_is_contiguous#_CPPv4NK18span_is_contiguousEv).
"""
function span_is_contiguous(@nospecialize(v::View))
    return DynamicCompilation.@compile_and_call(span_is_contiguous, (v,),
        compile_view(typeof(v); for_function=span_is_contiguous, no_error=true)
    )
end


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
function deep_copy(dest::View, src::View)
    @nospecialize dest src
    return DynamicCompilation.@compile_and_call(deep_copy, (dest, src), begin
        compile_view.((typeof(dest), typeof(src)); for_function=deep_copy, no_error=true)

        # Requirements in accordance with: https://kokkos.github.io/kokkos-core-wiki/API/core/view/deep_copy.html#requirements
        if eltype(src) != eltype(dest)
            error("`deep_copy` can only be used on Views with the same type: \
                   src=$(eltype(src)), dest=$(eltype(dest))")
        end

        if ndims(src) != ndims(dest)
            error("`deep_copy` can only be used on Views with the same number of dimensions: \
                   src=$(ndims(src)), dest=$(ndims(dest))")
        end

        view_types, view_dims, view_layouts, mem_spaces = _extract_view_params(typeof(src))
        _, _, dest_layouts, dest_mem_spaces = _extract_view_params(typeof(dest))

        DynamicCompilation.compile_and_load(@__MODULE__, "copy";
            view_types, view_dims, view_layouts,
            mem_spaces, dest_layouts, dest_mem_spaces,
            without_exec_space_arg = true,
        )
    end)
end


function deep_copy(space::ExecutionSpace, dest::View, src::View)
    @nospecialize space dest src
    return DynamicCompilation.@compile_and_call(deep_copy, (dest, src), begin
        compile_view.((typeof(dest), typeof(src)); for_function=deep_copy, no_error=true)

        # Requirements in accordance with: https://kokkos.github.io/kokkos-core-wiki/API/core/view/deep_copy.html#requirements
        if eltype(src) != eltype(dest)
            error("`deep_copy` can only be used on Views with the same type: \
                src=$(eltype(src)), dest=$(eltype(dest))")
        end

        if ndims(src) != ndims(dest)
            error("`deep_copy` can only be used on Views with the same number of dimensions: \
                src=$(ndims(src)), dest=$(ndims(dest))")
        end

        view_types, view_dims, view_layouts, mem_spaces = _extract_view_params(typeof(src))
        _, _, dest_layouts, dest_mem_spaces = _extract_view_params(typeof(dest))

        exec_spaces = [typeof(space)]
        DynamicCompilation.compile_and_load(@__MODULE__, "copy";
            view_types, view_dims, view_layouts,
            mem_spaces, exec_spaces, dest_layouts, dest_mem_spaces,
            without_exec_space_arg = false
        )
    end)
end


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


function create_mirror(src::View, mem_space, zero_fill)
    @nospecialize src mem_space zero_fill
    return DynamicCompilation.@compile_and_call(create_mirror, (src, mem_space, zero_fill), begin
        compile_view(typeof(src); for_function=create_mirror, no_error=true)
        view_types, view_dims, view_layouts, mem_spaces = _extract_view_params(typeof(src))
        dest_mem_spaces = isnothing(mem_space) ? DataType[] : [typeof(mem_space)]
        DynamicCompilation.compile_and_load(@__MODULE__, "mirrors";
            view_types, view_dims, view_layouts,
            mem_spaces, dest_mem_spaces,
            with_nothing_arg = isnothing(mem_space)
        )
    end)
end


"""
    create_mirror_view(src::View; mem_space = nothing, zero_fill = false)

Equivalent to [`create_mirror`](@ref), but if `src` is already accessible by the host, `src` is
returned and no view is created.
"""
function create_mirror_view(src::View; mem_space = nothing, zero_fill = false)
    return create_mirror_view(src, mem_space, zero_fill)
end


function create_mirror_view(src::View, mem_space, zero_fill)
    @nospecialize src mem_space zero_fill
    return DynamicCompilation.@compile_and_call(
            create_mirror_view, (src, mem_space, zero_fill), begin
        compile_view(typeof(src); for_function=create_mirror_view, no_error=true)
        view_types, view_dims, view_layouts, mem_spaces = _extract_view_params(typeof(src))
        dest_mem_spaces = isnothing(mem_space) ? DataType[] : [typeof(mem_space)]
        DynamicCompilation.compile_and_load(@__MODULE__, "mirrors";
            view_types, view_dims, view_layouts,
            mem_spaces, dest_mem_spaces,
            with_nothing_arg = isnothing(mem_space)
        )
    end)
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
4×4 Kokkos.Wrapper.Impl.View2D_R_HostAllocated{Float64}:
 1.0  5.0   9.0  13.0
 2.0  6.0  10.0  14.0
 3.0  7.0  11.0  15.0
 4.0  8.0  12.0  16.0

julia> Kokkos.subview(v, (2:3, 2:3))
2×2 Kokkos.Wrapper.Impl.View2D_R_HostAllocated{Float64}:
 6.0  10.0
 7.0  11.0

julia> Kokkos.subview(v, (:, 1))  # The subview may change its layout to `LayoutStride` 
4-element Kokkos.Wrapper.Impl.View1D_S_HostAllocated{Float64}:
 1.0
 2.0
 3.0
 4.0

julia> Kokkos.subview(v, (1,))  # Equivalent to `(1, :)`
4-element Kokkos.Wrapper.Impl.View1D_R_HostAllocated{Float64}:
  1.0
  5.0
  9.0
 13.0
```

# !!! warning

    `Kokkos.subview` is __not__ equivalent to `Base.view`, as it returns a new `Kokkos.View` object,
    while `Base.view` returns a `SubArray`, which cannot be passed to a `ccall`.

"""
function subview(view::View, indexes::Tuple, subview_dim::Type{<:Val}, subview_layout::Type)
    @nospecialize view indexes subview_dim subview_layout
    return DynamicCompilation.@compile_and_call(
            subview, (view, indexes, subview_dim, subview_layout), begin
        view_t = typeof(view)
        sub_dim = first(subview_dim.parameters)

        sub_view_t = View{eltype(view_t), sub_dim, array_layout(view_t), memory_space(view_t)}
        compile_view.((view_t, sub_view_t); for_function=subview, no_error=true)

        if (array_layout(view_t) != LayoutStride)
            # We also need the strided version of the subview type since we are compiling for all
            # `Kokkos::subview` instantiations which results in a view with `sub_dim` dimensions.
            sub_view_strided = View{eltype(view_t), sub_dim, LayoutStride, memory_space(view_t)}
            compile_view(sub_view_strided; for_function=subview, no_error=true)
        end

        view_types, view_dims, view_layouts, mem_spaces = _extract_view_params(view_t)
        subview_dims = [sub_dim]
        DynamicCompilation.compile_and_load(@__MODULE__, "subviews";
            view_types, view_dims, view_layouts, mem_spaces, subview_dims
        )
    end)
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


function subview(
    v::View{T, D, L, S},
    indexes::Tuple{Vararg{Union{Int, Colon, AbstractUnitRange}}}
) where {T, D, L, S}
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
    It is the responsibility of the user to make sure the original array is kept alive as long as
    the view should be accessed.
"""
function view_wrap(view_t::Type{View{T, D, L, S}}, d::Dims{D}, layout, p::Ptr{T}) where {T, D, L, S}
    @nospecialize view_t d layout p
    return DynamicCompilation.@compile_and_call(view_wrap, (view_t, d, layout, p), begin
        compile_view(view_t; for_function=view_wrap, no_error=true) 
    end)
end


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

@inline Base.size(v::View) = _get_dims(v)

@inline to_c_index(I::Vararg{Int, D}) where {D} = convert.(Idx, I .- 1)
Base.@propagate_inbounds elem_ptr(v::View{T, D}, I::Vararg{Int, D}) where {T, D} =
    (@boundscheck checkbounds(v, I...); _get_ptr(v, to_c_index(I...)...))

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

Base.strides(v::View) = _get_strides(v)

Base.unsafe_convert(::Type{Ptr{T}}, v::V) where {T, V <: View{T}} = pointer(v)

Base.elsize(::Type{<:View{T}}) where {T} = sizeof(T)

end
