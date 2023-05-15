module Spaces

using CxxWrap
import ..Kokkos: ensure_kokkos_wrapper_loaded, fence, get_impl_module

export Space, ExecutionSpace, MemorySpace
export Serial, OpenMP, OpenACC, OpenMPTarget, Threads, Cuda, HIP, HPX, SYCL
export HostSpace, CudaSpace, CudaHostPinnedSpace, CudaUVMSpace, HIPSpace, HIPHostPinnedSpace, HIPManagedSpace
export COMPILED_EXEC_SPACES, COMPILED_MEM_SPACES
export DEFAULT_DEVICE_SPACE, DEFAULT_HOST_SPACE, DEFAULT_DEVICE_MEM_SPACE, DEFAULT_HOST_MEM_SPACE
export SHARED_MEMORY_SPACE, SHARED_HOST_PINNED_MEMORY_SPACE
export memory_space, execution_space, enabled, kokkos_name
export accessible, array_layout, main_space_type, impl_space_type
export fence, concurrency, allocate, deallocate


"""
    Space

Abstract super type of all execution and memory spaces.

Main subtypes:
 - `ExecutionSpaces`: super type of all execution spaces
 - `MemorySpaces`: super type of all memory spaces

All Kokkos spaces have a main abstract type (`Serial`, `Cuda`, `HostSpace`, `HIPSpace`...) which are
defined even if it has not been compiled on the C++ side. Those main abstract types should be the
ones used when specifying a space. This allows methods like `enabled` to work independently from the
compiled internal library.

When a space is compiled, a sub-type of its main type is defined by `CxxWrap`, leading to the
following type structure:
`SerialImplAllocated <: SerialImpl <: Serial <: ExecutionSpace <: Space`.
Below the main space type (here, `Serial`), the sub-types are only defined if they are compiled, and
therefore they should not be relied upon. 

Navigating the type tree can be made easier through [`main_space_type`](@ref).
"""
abstract type Space end


"""
    ExecutionSpace <: Space

Abstract super-type of all execution spaces.

Sub-types:
 - `Serial`
 - `OpenMP`
 - `OpenACC`
 - `OpenMPTarget`
 - `Threads`
 - `Cuda`
 - `HIP`
 - `HPX`
 - `SYCL`

All sub-types are always defined, but only some of them are [`enabled`](@ref).
To enable an execution space, you must enable its related Kokkos backend, e.g.
`"-DKokkos_ENABLE_SERIAL=ON"` for the `Serial` execution space.

To do this you can set the [backends](@ref) option with `Kokkos.set_backends`,
or specify the option directly through [kokkos_options](@ref).
"""
abstract type ExecutionSpace <: Space          end
abstract type Serial         <: ExecutionSpace end
abstract type OpenMP         <: ExecutionSpace end
abstract type OpenACC        <: ExecutionSpace end
abstract type OpenMPTarget   <: ExecutionSpace end
abstract type Threads        <: ExecutionSpace end
abstract type Cuda           <: ExecutionSpace end
abstract type HIP            <: ExecutionSpace end
abstract type HPX            <: ExecutionSpace end
abstract type SYCL           <: ExecutionSpace end


const ALL_BACKENDS = [Serial, OpenMP, OpenACC, OpenMPTarget, Threads, Cuda, HIP, HPX, SYCL]


"""
    MemorySpace <: Space

Abstract super-type of all memory spaces.

Sub-types:
 - `HostSpace`
 - `CudaSpace`
 - `CudaHostPinnedSpace`
 - `CudaUVMSpace`
 - `HIPSpace`
 - `HIPHostPinnedSpace`
 - `HIPManagedSpace`

Sub-types work the same as for [`ExecutionSpace`](@ref). They can be enabled by enabling their
respective backend.
"""
abstract type MemorySpace           <: Space       end
abstract type HostSpace             <: MemorySpace end
abstract type CudaSpace             <: MemorySpace end
abstract type CudaHostPinnedSpace   <: MemorySpace end
abstract type CudaUVMSpace          <: MemorySpace end
abstract type HIPSpace              <: MemorySpace end
abstract type HIPHostPinnedSpace    <: MemorySpace end
abstract type HIPManagedSpace       <: MemorySpace end


const ALL_MEM_SPACES = [HostSpace, CudaSpace, CudaHostPinnedSpace, CudaUVMSpace, HIPSpace, HIPHostPinnedSpace, HIPManagedSpace]


# Defined in 'spaces.cpp', in 'post_register_space'
"""
    execution_space(space::Union{<:MemorySpace, Type{<:MemorySpace}})

Return the execution space associated by default to the given memory space.
"""
execution_space(::S) where {S <: Space} = execution_space(main_space_type(S))
execution_space(::Type{S}) where {S <: ExecutionSpace} = error("space $S must be a memory space")
execution_space(::Type{S}) where {S <: MemorySpace} = error("space $S is not compiled")


# Defined in 'spaces.cpp', in 'post_register_space'
"""
    memory_space(space::Union{<:ExecutionSpace, Type{<:ExecutionSpace}})

Return the memory space associated by default to the given execution space.
"""
memory_space(::S) where {S <: Space} = memory_space(main_space_type(S))
memory_space(::Type{S}) where {S <: MemorySpace} = error("space $S must be an execution space")
memory_space(::Type{S}) where {S <: ExecutionSpace} = error("space $S is not compiled")


# Defined in 'spaces.cpp', in 'register_space'
"""
    enabled(space::Union{Space, Type{<:Space}})

Return true if the given execution or memory space is compiled.
"""
enabled(::S) where {S <: Space} = enabled(main_space_type(S))
enabled(::Type{<:Space}) = false


# Defined in 'spaces.cpp', in 'register_space'
"""
    kokkos_name(space::Union{Space, Type{<:Space}})

Return the name of the execution or memory space as defined by Kokkos.

Equivalent to `Kokkos::space::name()`
"""
kokkos_name(::Type{<:Space}) = error("space $S is not compiled")


# Defined in 'spaces.cpp', in 'post_register_all'
"""
    accessible([S1::Union{<:Space, Type{<:Space}},] S2::Union{<:MemorySpace, Type{<:MemorySpace}})

Return `true` if the memory space `S2` is accessible from `S1` (a memory or execution space).

If only `S2` is specified, `S1` defaults to [`DEFAULT_HOST_SPACE`](@ref).

Equivalent to [`Kokkos::SpaceAccessibility<S1, S2>::accessible`](https://kokkos.github.io/kokkos-core-wiki/API/core/SpaceAccessibility.html)
"""
accessible(::S1, ::S2) where {S1 <: Space, S2 <: MemorySpace} = accessible(S1, S2)
accessible(::Type{S1}, ::S2) where {S1 <: Space, S2 <: MemorySpace} = accessible(S1, S2)
accessible(::S1, ::Type{S2}) where {S1 <: Space, S2 <: MemorySpace} = accessible(S1, S2)

function accessible(S1::Type{<:Space}, S2::Type{<:MemorySpace})
    main_S1_type = main_space_type(S1)
    main_S2_type = main_space_type(S2)
    return accessible(main_S1_type, main_S2_type)
end

function accessible(space::Union{<:MemorySpace, Type{<:MemorySpace}})
    if DEFAULT_HOST_SPACE === nothing
        ensure_kokkos_wrapper_loaded()
    end
    return accessible(DEFAULT_HOST_SPACE, space)
end


# Defined in 'spaces.cpp', in 'post_register_space'
"""
    array_layout(exec_space::Union{<:ExecutionSpace, Type{<:ExecutionSpace}})

Return the default array layout type of the given execution space.
"""
array_layout(::S) where {S <: Space} = array_layout(S)

function array_layout(S::Type{<:ExecutionSpace})
    main_S_type = main_space_type(S)
    main_S_type !== S && return array_layout(main_S_type)
    error("execution space $S is not compiled")
end


"""
    main_space_type(space::Union{<:Space, Type{<:Space}})

Return the main space type of `space`, e.g. for `Serial`, `SerialImpl` or `SerialImplAllocated` we
get `Serial`.
"""
function main_space_type(::Type{S}) where {S <: Space}
    if (S == Space || S == MemorySpace || S == ExecutionSpace)
        error("$S must be a subtype of `MemorySpace` or `ExecutionSpace`")
    end
    mt = S
    while (mt.super != MemorySpace) && (mt.super != ExecutionSpace)
        mt = mt.super
    end
    return mt
end

main_space_type(::S) where {S <: Space} = main_space_type(S)


"""
    impl_space_type(::Type{<:Space})

Opposite of [`main_space_type`](@ref): from the main space type (`Serial`, `OpenMP`, `HostSpace`...)
return the implementation type (`SerialImpl`, `OpenMPImpl`, `HostSpaceImpl`...).
The given space must be compiled.
"""
impl_space_type(::Type{S}) where {S <: Space} = error("space $S is not compiled")


# Space constructors
for S_type in (Serial, OpenMP, OpenACC, OpenMPTarget, Threads, Cuda, HIP, HPX, SYCL,
               HostSpace, CudaSpace, CudaHostPinnedSpace, CudaUVMSpace,
               HIPSpace, HIPHostPinnedSpace, HIPManagedSpace)
    @eval (::Type{$S_type})() = impl_space_type($S_type)()
end


# Defined in 'spaces.cpp', in 'register_space'
"""
    fence(exec_space::ExecutionSpace)

Wait for all asynchronous tasks operating on this execution space instance to complete.

Equivalent to [`exec_space.fence()`](https://kokkos.github.io/kokkos-core-wiki/API/core/execution_spaces.html#functionality).
"""
function fence(::ExecutionSpace) end


# Defined in 'spaces.cpp', in 'register_space'
"""
    concurrency(exec_space::ExecutionSpace)

The maximum number of threads utilized by the execution space instance.

Equivalent to [`exec_space.concurrency()`](https://kokkos.github.io/kokkos-core-wiki/API/core/execution_spaces.html#functionality).
"""
function concurrency end


# Defined in 'spaces.cpp', in 'register_space'
"""
    allocate(mem_space::MemorySpace, bytes)

Allocate `bytes` on the memory space instance. Returns a pointer to the allocated memory.

Equivalent to [`mem_space.allocate(bytes)`](https://kokkos.github.io/kokkos-core-wiki/API/core/memory_spaces.html#functions)
"""
function allocate end


# Defined in 'spaces.cpp', in 'register_space'
"""
    deallocate(mem_space::MemorySpace, ptr)

Frees `ptr`, previously allocated with [`allocate`](@ref).

Equivalent to [`mem_space.deallocate(ptr, bytes)`](https://kokkos.github.io/kokkos-core-wiki/API/core/memory_spaces.html#functions)
"""
function deallocate end


# Defined in 'spaces.cpp', in 'register_all'
"""
    COMPILED_EXEC_SPACES::Tuple{Vararg{Type{<:ExecutionSpace}}}

List of all compiled Kokkos execution spaces.

`nothing` if Kokkos is not yet loaded.
"""
COMPILED_EXEC_SPACES = nothing


# Defined in 'spaces.cpp', in 'define_execution_spaces_functions'
"""
    DEFAULT_DEVICE_SPACE::Type{<:ExecutionSpace}

The default execution space where kernels are applied on the device.

Equivalent to `Kokkos::DefaultExecutionSpace`.

`nothing` if Kokkos is not yet loaded.
"""
DEFAULT_DEVICE_SPACE = nothing


# Defined in 'spaces.cpp', in 'define_execution_spaces_functions'
"""
    DEFAULT_HOST_SPACE::Type{<:ExecutionSpace}

The default execution space where kernels are applied on the host.

Equivalent to `Kokkos::DefaultHostExecutionSpace`.

`nothing` if Kokkos is not yet loaded.
"""
DEFAULT_HOST_SPACE = nothing


# Defined in 'spaces.cpp', in 'register_all'
"""
    COMPILED_MEM_SPACES::Tuple{Vararg{Type{<:MemorySpace}}}

List of all compiled Kokkos execution spaces.

`nothing` if Kokkos is not yet loaded.
"""
COMPILED_MEM_SPACES = nothing


# Defined in 'spaces.cpp', in 'define_memory_spaces_functions'
"""
    DEFAULT_DEVICE_MEM_SPACE::Type{<:MemorySpace}

The default memory space where views are stored on the device.

Equivalent to `Kokkos::DefaultExecutionSpace::memory_space`.

`nothing` if Kokkos is not yet loaded.
"""
DEFAULT_DEVICE_MEM_SPACE = nothing


# Defined in 'spaces.cpp', in 'define_memory_spaces_functions'
"""
    DEFAULT_HOST_MEM_SPACE::Type{<:MemorySpace}

The default memory space where views are stored on the host.

Equivalent to `Kokkos::DefaultHostExecutionSpace::memory_space`.

`nothing` if Kokkos is not yet loaded.
"""
DEFAULT_HOST_MEM_SPACE = nothing


# Defined in 'spaces.cpp', in 'define_memory_spaces_functions'
"""
    SHARED_MEMORY_SPACE::Union{Nothing, Type{<:MemorySpace}}

The shared memory space between the host and device, or `nothing` if there is none.

Equivalent to `Kokkos::SharedSpace` if `Kokkos::has_shared_space` is `true`.

`nothing` if Kokkos is not yet loaded.
"""
SHARED_MEMORY_SPACE = nothing


# Defined in 'spaces.cpp', in 'define_memory_spaces_functions'
"""
    SHARED_HOST_PINNED_MEMORY_SPACE::Union{Nothing, Type{<:MemorySpace}}

The shared pinned memory space between the host and device, or `nothing` if there is none.

Equivalent to `Kokkos::SharedHostPinnedSpace` if `Kokkos::has_shared_host_pinned_space` is `true`.

`nothing` if Kokkos is not yet loaded.
"""
SHARED_HOST_PINNED_MEMORY_SPACE = nothing


module BackendFunctions

# OpenMP

"""
    omp_set_num_threads(threads::Cint)::Cvoid
   
[See the OpenMP docs](https://www.openmp.org/spec-html/5.0/openmpsu110.html)
"""
function omp_set_num_threads end

"""
    omp_get_max_threads()::Cint
   
[See the OpenMP docs](https://www.openmp.org/spec-html/5.0/openmpsu112.html)
"""
function omp_get_max_threads end

"""
    omp_get_proc_bind()::Cint
   
[See the OpenMP docs](https://www.openmp.org/spec-html/5.0/openmpsu132.html)
"""
function omp_get_proc_bind end

"""
    omp_get_num_places()::Cint

[See the OpenMP docs](https://www.openmp.org/spec-html/5.0/openmpsu133.html)
"""
function omp_get_num_places end

"""
    omp_get_place_num_procs(place::Cint)::Cint

[See the OpenMP docs](https://www.openmp.org/spec-html/5.0/openmpsu134.html)
"""
function omp_get_place_num_procs end

"""
    omp_get_place_proc_ids(place::Cint)::Vector{Cint}

[See the OpenMP docs](https://www.openmp.org/spec-html/5.0/openmpsu135.html)
"""
function omp_get_place_proc_ids end

"""
    omp_capture_affinity([format::String])::String

More or less equivalent to 
[`omp_display_affinity(char*)`](https://www.openmp.org/spec-html/5.0/openmpsu141.html#x178-8280003.2.32),
but applies the given `format` (or OpenMP's default one) to each OpenMP thread using 
[`omp_capture_affinity`](https://www.openmp.org/spec-html/5.0/openmpsu142.html#x179-8340003.2.33),
and returns the result.
"""
function omp_capture_affinity end

end


function __init_vars()
    impl = get_impl_module()
    global COMPILED_EXEC_SPACES = Base.invokelatest(impl.__compiled_exec_spaces)
    global DEFAULT_DEVICE_SPACE = Base.invokelatest(impl.__default_device_space)
    global DEFAULT_HOST_SPACE = Base.invokelatest(impl.__default_host_space)
    global COMPILED_MEM_SPACES = Base.invokelatest(impl.__compiled_mem_spaces)
    global DEFAULT_DEVICE_MEM_SPACE = Base.invokelatest(impl.__default_memory_space)
    global DEFAULT_HOST_MEM_SPACE = Base.invokelatest(impl.__default_host_memory_space)
    global SHARED_MEMORY_SPACE = Base.invokelatest(impl.__shared_memory_space)
    global SHARED_HOST_PINNED_MEMORY_SPACE = Base.invokelatest(impl.__shared_host_pinned_space)
end

end