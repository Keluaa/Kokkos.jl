```@meta
CurrentModule = Kokkos
```

# Using views in an inaccessible memory space

By Kokkos semantics, it is not possible to read or write views which are stored in an inaccessible
memory space ([`accessible(view) == false`](@ref accessible)).
Therefore views on a GPU device (stored in a `CudaSpace`, `HIPSpace`, etc...) cannot be displayed:

```julia-repl
julia> v = Kokkos.View{Float64}(10; mem_space=Kokkos.CudaSpace)
10-element Kokkos.Views.View{Float64, 1, Kokkos.LayoutLeft, Kokkos.CudaSpace}: <inaccessible view>
```

This can be viewed as a stricter version of `allowscalar(false)` in CUDA.jl or AMDGPU.jl.

The correct approch is the same as in C++: creating a host copy of the view.
```julia-repl
julia> host_v = Kokkos.create_mirror_view(v; mem_space=Kokkos.HostSpace());

julia> copyto!(host_v, v)  # Calls `Kokkos.deep_copy`

julia> host_v
10-element Kokkos.Views.View{Float64, 1, Kokkos.LayoutLeft, Kokkos.HostSpace}:
 0.0
 0.0
 0.0
...

julia> host_v[1] = 1
1

julia> copyto!(v, host_v)

```

The strength of this approach is the fact that it will be efficient whatever the device backend is.
[`create_mirror_view`](@ref) will simply return the `view` passed to it if the memory space is the
same, making it effectively a no-op.
Same for [`deep_copy`](@ref), which is a no-op if both arguments are the same.
