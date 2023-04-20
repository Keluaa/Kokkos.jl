```@meta
CurrentModule = Kokkos
```

# Using Kokkos.jl with MPI.jl


## Loading and compilation

Since calling [`Kokkos.initialize`](@ref) may trigger the compilation of the internal wrapper
library, some care is needed to make sure only a single process is compiling.

A basic initialization workflow with MPI may look like this:
```julia
using MPI
using Kokkos

MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    Kokkos.load_wrapper_lib()  # All compilation (if any) of the C++ wrapper happens here
end

MPI.Barrier(MPI.COMM_WORLD)

rank != 0 && Kokkos.load_wrapper_lib(; no_compilation=true, no_git=true)
Kokkos.initialize()
```

Note that passing `no_compilation=true` and `no_git=true` to [`load_wrapper_lib`](@ref) on the
non-root processes is required.

If configuration options need to be changed before initializing Kokkos, then it is preferable to
perform the changes on the root process before `using Kokkos` is called on the others, since
changing options modify the `LocalPreferrences.toml` file.


## Passing views to MPI

Passing a [`Kokkos.View`](@ref) to a MPI directive is possible:

```julia
v = View{Float64}(n)
v .= MPI.Comm_rank(MPI.COMM_WORLD)

r = View{Float64}(n)

MPI.Sendrecv!(v, next_rank, 0, r, prev_rank, 0, MPI.COMM_WORLD)

@assert all(r .== prev_rank)
```

Internally, the pointer to the data of the view is passed to MPI, there is no copy of the data.
The data length given to MPI is not `length(view)` but
[`Kokkos.memory_span(view)`](@ref Kokkos.memory_span), therefore views with an irregular layout will
work as long as their data is stored in a single memory block.

Support for GPU-awareness should be seamless, as long as your MPI implementation supports the GPU.
