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

The same workflow can be used to compile your library on the root process:

```julia
my_project = CMakeKokkosProject(project_src, "libproj")
rank == 0 && compile(my_project)
MPI.Barrier(MPI.COMM_WORLD)
lib = load_lib(my_project)
```

If configuration options need to be changed before initializing Kokkos, then it is preferable to
perform the changes on the root process before `using Kokkos` is called on the others, since
changing options will modify the `LocalPreferrences.toml` file.


## Passing views to MPI

Passing a [`Kokkos.View`](@ref) to a MPI directive is possible:

```julia
v = View{Float64}(n)
v .= MPI.Comm_rank(MPI.COMM_WORLD)

r = View{Float64}(n)

MPI.Sendrecv!(v, next_rank, 0, r, prev_rank, 0, MPI.COMM_WORLD)

@assert all(r .== prev_rank)
```

Internally, the pointer to the data of the view is passed to MPI, there is no copy of the data,
regardless of the memory space where the view is stored in.

If `Kokkos.span_is_contiguous(view) == true`, then the whole memory span of the view is passed to
MPI as a single block of data.

For non-contiguous views (such as `LayoutStride`), a custom `MPI.Datatype` is built to exactly
represent the view.

Support for GPU-awareness should be seamless, as long as your MPI implementation supports the GPU.
