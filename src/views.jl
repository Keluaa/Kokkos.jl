module Views

using CxxWrap
import ..Kokkos: ExecutionSpace, MemorySpace
import ..Kokkos: KOKKOS_LIB_PATH, COMPILED_MEM_SPACES, DEFAULT_DEVICE_MEM_SPACE
import ..Kokkos: memory_space, accessible, main_space_type

export View, ViewPtr, Idx
export COMPILED_TYPES, COMPILED_DIMS
export label


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
"""
abstract type View{T, D, MemSpace} <: Base.AbstractArray{T, D} end


@wrapmodule(KOKKOS_LIB_PATH, :define_kokkos_views)

function __init__()
    @initcxx
end


"""
    Idx::Type{<:Integer}

Integer type used by views for indexing on the default execution device. Usually either `Cint` or
`Clonglong`.

Equivalent to `Kokkos::RangePolicy<>::index_type`.
"""
const Idx = idx_type()


"""
    COMPILED_DIMS::Tuple{Vararg{Int32}}

List of all View dimensions which are compiled.

By default, dimensions 1 and 2 are compiled.
"""
const COMPILED_DIMS = compiled_dims()


"""
    COMPILED_TYPES::Tuple{Vararg{DataType}}

List of all View element types which are compiled

By default, the following types are compiled: Float64 (double), Float32 (float) and Int64 (int64_t).
"""
const COMPILED_TYPES = compiled_types()


"""
    alloc_view(::Type{T}, dims::Union{Integer, Tuple{Vararg{<:Integer}}};
        mem_space=DEFAULT_DEVICE_MEM_SPACE,
        label="",
        zero_fill=false,
        dim_pad=false
    )

Allocate a new `Kokkos::View` of type `T` with the given `dims` in the memory space `mem_space`.

`dims` is either an integer (1D), or a tuple of integers.

Optional arguments:
 - `mem_space` is the memory space of the view, a `Kokkos.ExecutionSpace`. It is by default the
   memory space of the device: [`DEFAULT_DEVICE_MEM_SPACE`](@ref). It can be given as a type: then a
   new memory space will be default-constructed.
 - `label` is the debug label of the view.
 - `zero_fill` controls the initialization of the view. If `true`, all elements will be set to `0`.
   Uses `Kokkos::WithoutInitializing` internally.
 - `dim_pad` controls the padding of dimensions. Uses `Kokkos::AllowPadding` internally.

See [the Kokkos documentation about `Kokkos::view_alloc()`](https://kokkos.github.io/kokkos-core-wiki/API/core/view/view_alloc.html)
"""
function alloc_view end

# Instances of `alloc_view` for each compiled type, dimension and memory space are defined from the C++ library

function alloc_view(::Type{T}, ::Val{D}, ::Type{S}, dims, mem_space, label, zero_fill, dim_pad) where {T, D, S}
    # Fallback: error handler
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


function alloc_view(::Type{T}, D::Int, dims;
    mem_space = DEFAULT_DEVICE_MEM_SPACE,
    label = "",
    zero_fill = false,
    dim_pad = false
) where {T}
    if mem_space isa DataType
        # Default construction of the memory space
        if mem_space <: ExecutionSpace
            return alloc_view(T, Val(D), memory_space(mem_space), dims, nothing, label, zero_fill, dim_pad)
        else
            return alloc_view(T, Val(D), mem_space, dims, nothing, label, zero_fill, dim_pad)
        end
    else
        # `mem_space` is an instance of a memory space
        return alloc_view(T, Val(D), typeof(mem_space), dims, mem_space, label, zero_fill, dim_pad)
    end
end


alloc_view(::Type{T}, dims; kwargs...) where {T} = alloc_view(T, length(dims), dims; kwargs...)


"""
    accessible(::View)

Return `true` if the view is accesssible from the default host execution space.
"""
accessible(::View{T, D, MemSpace}) where {T, D, MemSpace} = accessible(MemSpace)


"""
    label(::View)

Return the label of the `View`.
"""
function label end


# === Constructors ===

"""
    View{T}(undef, dims; kwargs...)
    View{T}(dims; kwargs...)

Construct an N-dimensional `View{T}`. `dims` is either an integer or a tuple of integers.

See [`alloc_view`](@ref) for the documentation about the `kwargs`, allowing to specify the memory
space or label of the view.

If `undef` is not given, then `zero_fill=true` is passed to [`alloc_view`](@ref).
"""
View{T}(::UndefInitializer, dims; kwargs...) where T = alloc_view(T, length(dims), dims; kwargs...)
View{T}(dims; kwargs...) where T = alloc_view(T, length(dims), dims; zero_fill = true, kwargs...)

# TODO: better constructors which cover most combinasions, see CUDA.jl: https://github.com/JuliaGPU/CUDA.jl/blob/master/src/array.jl#L176 


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


# TODO: copy 'dim_pad' too
Base.similar(a::View{T, D, M}) where {T, D, M} =
    alloc_view(T, D, size(a); zero_fill=false, mem_space=main_space_type(M))
Base.similar(::View{T, <: Any, M}, dims::Dims{D}) where {T, D, M} =
    alloc_view(T, D, dims; zero_fill=false, mem_space=main_space_type(M))
Base.similar(::View{<: Any,<: Any, M}, ::Type{T}, dims::Dims{D}) where {T, D, M} =
    alloc_view(T, D, dims; zero_fill=false, mem_space=main_space_type(M))


# === Pointer conversion ===


# TODO: make this an UnionAll, to be able to write `ViewPtr{Float64, 2}` or `ViewPtr{Float64, 1, HostSpace}`
const ViewPtr = Ptr{Nothing}

Base.unsafe_convert(::Type{ViewPtr}, v::View) = v.cpp_object

end