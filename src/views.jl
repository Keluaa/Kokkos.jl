module Views

using CxxWrap
import ..Kokkos: ExecutionSpace, MemorySpace
import ..Kokkos: COMPILED_MEM_SPACES, DEFAULT_DEVICE_MEM_SPACE, DEFAULT_HOST_MEM_SPACE
import ..Kokkos: ensure_kokkos_wrapper_loaded, get_impl_module
import ..Kokkos: memory_space, accessible, main_space_type, finalize

export View, Idx
export COMPILED_TYPES, COMPILED_DIMS
export label, view_wrap, view_data, memory_span


# The View type must be defined before loading the Kokkos wrapper library which depends on it
"""
    View{T, D, MemSpace} <: AbstractArray{T, D}

Wrapper around a `Kokkos::View` of `D` dimensions of type `T`, stored in `MemSpace`.

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
abstract type View{T, D, MemSpace} <: Base.AbstractArray{T, D} end


# Internal functions, defined for each view in 'views.cpp', in 'register_view_types'
function get_ptr end
function get_dims end


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

By default, the following types are compiled: Float64 (double), Float32 (float) and Int64 (int64_t).

`nothing` if Kokkos is not yet loaded.
"""
COMPILED_TYPES = nothing


function error_view_not_compiled(::Type{View{T, D, S}}) where {T, D, S}
    ensure_kokkos_wrapper_loaded()

    if !(D in COMPILED_DIMS)
        dims_str = join(string.(COMPILED_DIMS) .* 'D', ", ", " and ")
        pluralized_str = length(COMPILED_DIMS) > 1 ? "are" : "is"
        error("`Kokkos.View$(D)D` cannot be created, as this dimension was not compiled. \
                Only $dims_str $pluralized_str compiled.")
    end

    if !(T in COMPILED_TYPES)
        types_str = join(COMPILED_TYPES, ", ", " and ")
        pluralized_str = length(COMPILED_TYPES) > 1 ? "are" : "is"
        error("view type `$T` is not compiled. Only $types_str $pluralized_str compiled.")
    end

    if !(S in COMPILED_MEM_SPACES)
        spaces_str = join(COMPILED_MEM_SPACES, ", ", " and ")
        pluralized_str = length(COMPILED_MEM_SPACES) > 1 ? "spaces are" : "space is"
        error("memory space `$S` is not compiled. The only compiled $pluralized_str $spaces_str.")
    end

    error("$(D)D views of type $T stored in $S are not compiled.")
end


# Instances of `alloc_view` for each compiled type, dimension and memory space are defined in
# 'views.cpp', in 'register_constructor'.
function alloc_view(::Type{View{T, D, S}}, dims::Dims{D}, mem_space, label, zero_fill, dim_pad) where {T, D, S}
    # Fallback: error handler
    error_view_not_compiled(View{T, D, S})
end


"""
    accessible(::View)

Return `true` if the view is accessible from the default host execution space.
"""
accessible(::View{T, D, MemSpace}) where {T, D, MemSpace} = accessible(MemSpace)


"""
    memory_space(::View)

The memory space type in which the view data is stored.

```julia-repl
julia> my_cuda_space = Kokkos.impl_space_type(Kokkos.CudaSpace)()
 ...

julia> v = View{Float64}(undef, 10; mem_space=my_cuda_space)
 ...

julia> memory_space(v)
Kokkos.Spaces.CudaSpace
```
"""
memory_space(::View{T, D, MemSpace}) where {T, D, MemSpace} = main_space_type(MemSpace)


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


# === Constructors ===

"""
    View{T, D, S}(dims;
        mem_space = DEFAULT_DEVICE_MEM_SPACE,
        label = "",
        zero_fill = true,
        dim_pad = false
    )

Construct an N-dimensional `View{T}`.

`D` can be deduced from `dims`, which can either be a `NTuple{D, Integer}` or `D` integers.

`S` defaults to the type of `mem_space`. `mem_space` can be an [`ExecutionSpace`](@ref), in which
case it is converted to a [`MemorySpace`](@ref) with [`memory_space`](@ref). `mem_space` can be
either an instance of a [`MemorySpace`](@ref) or one of the main types of memory spaces, in which
case an instance of a [`MemorySpace`](@ref) is default constructed (behaviour of Kokkos by default).

The `label` is the debug label of the view.

If `zero_fill=true`, all elements will be set to `0`. Uses `Kokkos::WithoutInitializing` internally.

`dim_pad` controls the padding of dimensions. Uses `Kokkos::AllowPadding` internally. If `true`,
then a view might not have a layout identical to a classic `Array`, for better memory alignment.

See [the Kokkos documentation about `Kokkos::view_alloc()`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/view_alloc.html)
for more info.
"""
function View{T, D, S}(dims::Dims{D};
    mem_space = DEFAULT_DEVICE_MEM_SPACE,
    label = "",
    zero_fill = true,
    dim_pad = false
) where {T, D, S}
    mem_space_t = mem_space isa DataType ? mem_space : typeof(mem_space)
    if !(main_space_type(mem_space_t) <: S)
        error("Conficting types for View constructor typing `$S` and `mem_space` kwarg: $mem_space \
               (a $(main_space_type(mem_space_t)))")
    end

    if mem_space isa DataType
        # Let `alloc_view` call the memory space constructor
        return alloc_view(View{T, D, S}, dims, nothing, label, zero_fill, dim_pad)
    else
        return alloc_view(View{T, D, S}, dims, mem_space, label, zero_fill, dim_pad)
    end
end

View{T, D, S}(::UndefInitializer, dims::Dims{D}; kwargs...) where {T, D, S} =
    View{T, D, S}(dims; kwargs..., zero_fill=false)


# View{T, D} to View{T, D, S}
function View{T, D}(dims::Dims{D};
    mem_space = DEFAULT_DEVICE_MEM_SPACE,
    label = "",
    zero_fill = true,
    dim_pad = false
) where {T, D}
    if mem_space isa DataType
        # Default construction of the memory space (delegated to `alloc_view`)
        (mem_space <: ExecutionSpace) && (mem_space = memory_space(mem_space))
        return View{T, D, mem_space}(dims; mem_space, label, zero_fill, dim_pad)
    elseif mem_space isa MemorySpace
        mem_space_t = main_space_type(typeof(mem_space))
        return View{T, D, mem_space_t}(dims; mem_space, label, zero_fill, dim_pad)
    else
        ensure_kokkos_wrapper_loaded()
        throw(TypeError(:View, "constructor", MemorySpace, mem_space))
    end
end

View{T, D}(::UndefInitializer, dims::Dims{D}; kwargs...) where {T, D} =
    View{T, D}(dims; kwargs..., zero_fill=false)


# Int... to Dims{D}
"""
    View{T, D, S}(undef, dims; kwargs...)

Construct an N-dimensional `View{T}`, without initialization of its elements.

Strictly equivalent to passing `zero_fill=false` to the `kwargs`.
"""
View{T, D, S}(::UndefInitializer, dims::Integer...; kwargs...) where {T, D, S} =
    View{T, D, S}(convert(Tuple{Vararg{Int}}, dims); kwargs..., zero_fill=false)
View{T, D}(::UndefInitializer, dims::Integer...; kwargs...) where {T, D} =
    View{T, D}(convert(Tuple{Vararg{Int}}, dims); kwargs..., zero_fill=false)
View{T, D, S}(dims::Integer...; kwargs...) where {T, D, S} =
    View{T, D, S}(convert(Tuple{Vararg{Int}}, dims); kwargs...)
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
View{T, D, S}(; kwargs...) where {T, D, S} =
    View{T, D, S}(ntuple(Returns(0), D); kwargs..., zero_fill=false)
View{T, D}(; kwargs...) where {T, D} =
    View{T, D}(ntuple(Returns(0), D); kwargs..., zero_fill=false)
View{T}(; kwargs...) where {T} =
    View{T}((0,); kwargs..., zero_fill=false)


# TODO: add an array layout parameter
# Instances of `view_wrap` for each compiled type, dimension and memory space are defined in
# 'views.cpp', in 'register_constructor'.
"""
    view_wrap(array::DenseArray{T, D})
    view_wrap(::Type{View{T, D, S}}, array::DenseArray{T, D})
    view_wrap(::Type{View{T, D, S}}, d::NTuple{D, Int}, p::Ptr{T})

Construct a new [`View`](@ref) from the data of a Julia-allocated array.
The returned view does not own the data, and no copy is made.

The memory space `S` defaults to [`DEFAULT_HOST_MEM_SPACE`](@ref).

!!! warning

    Julia arrays have a column-major layout.
    Kokkos can handle both row and column major array layouts, but it is imposed by the memory
    space. If the layouts don't match the view behaviour is unpredictable.
    This only affects 2D arrays and above.

!!! warning

    The returned view does not hold a reference to the original Julia array.
    It is the responsability of the user to make sure the original array is kept alive as long as
    the view should be accessed.
"""
view_wrap(::Type{View{T, D, S}}, ::Dims{D}, ::Ptr{T}) where {T, D, S} =
    error_view_not_compiled(View{T, D, S})
view_wrap(::Type{View{T, D, S}}, a::DenseArray{T, D}) where {T, D, S} =
    view_wrap(View{T, D, main_space_type(S)}, size(a), pointer(a))
view_wrap(::Type{View{T, D}}, a::DenseArray{T, D}) where {T, D} =
    view_wrap(View{T, D, DEFAULT_HOST_MEM_SPACE}, a)
view_wrap(::Type{View{T}}, a::DenseArray{T, D}) where {T, D} =
    view_wrap(View{T, D}, a)
view_wrap(a::DenseArray{T, D}) where {T, D} =
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


# TODO: copy 'dim_pad' too (and LayoutType in the future)
Base.similar(a::View{T, D, M}) where {T, D, M} =
    View{T, D, main_space_type(M)}(size(a); zero_fill=false)
Base.similar(::View{T, <:Any, M}, dims::Dims{D}) where {T, D, M} =
    View{T, D, main_space_type(M)}(dims; zero_fill=false)
Base.similar(::View{<:Any, <:Any, M}, ::Type{T}, dims::Dims{D}) where {T, D, M} =
    View{T, D, main_space_type(M)}(dims; zero_fill=false)


# === Pointer conversion ===

Base.pointer(v::V) where {T, V <: View{T}} = Ptr{T}(view_data(v).cpp_object)

Base.cconvert(::Type{Ref{V}}, v::V) where {V <: View} = Ptr{Nothing}(v.cpp_object)


function __init_vars()
    impl = get_impl_module()
    global Idx = Base.invokelatest(impl.__idx_type)
    global COMPILED_DIMS = Base.invokelatest(impl.__compiled_dims)
    global COMPILED_TYPES = Base.invokelatest(impl.__compiled_types)
end

end