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