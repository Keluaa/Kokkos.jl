```@meta
CurrentModule = Kokkos.Spaces
```

# Execution & Memory Spaces

```@docs
Space
ExecutionSpace
MemorySpace
accessible(::Space)
enabled
execution_space
memory_space(::Space)
main_space_type
impl_space_type
kokkos_name
fence(::ExecutionSpace)
concurrency
allocate
deallocate
```

## Constants

```@docs
COMPILED_EXEC_SPACES
COMPILED_MEM_SPACES
```

## Default spaces

```@docs
DEFAULT_DEVICE_SPACE
DEFAULT_DEVICE_MEM_SPACE
DEFAULT_HOST_SPACE
DEFAULT_HOST_MEM_SPACE
SHARED_MEMORY_SPACE
SHARED_HOST_PINNED_MEMORY_SPACE
```

## Backend-specific methods

Those unexported methods are defined in the `Kokkos.Spaces.BackendFunctions` module.
They have methods only if their respective backend is enabled and Kokkos is initialized.

### OpenMP

Some functions of the OpenMP runtime are made available through `Kokkos.jl`, mostly for debugging
purposes and tracking thread affinity.

```@docs
BackendFunctions.omp_set_num_threads
BackendFunctions.omp_get_max_threads
BackendFunctions.omp_get_proc_bind
BackendFunctions.omp_get_num_places
BackendFunctions.omp_get_place_num_procs
BackendFunctions.omp_get_place_proc_ids
BackendFunctions.omp_capture_affinity
```
