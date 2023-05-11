```@meta
CurrentModule = Kokkos
```

# Calling a Kokkos library


Suppose we want to wrap the following function:

```c++
// my_lib.cpp
#include "Kokkos_Core.hpp"

extern "C"
void fill_view(Kokkos::View<double*>& view, double value)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, view.size()),
    KOKKOS_LAMBDA(int i) {
        view[i] = value;
    });
}
```

It is important to understand that here the argument type `Kokkos::View<double*>&` relies on the
default template arguments for the layout type, memory space and memory traits.
Therefore, its complete type will change depending on the Kokkos configuration.

In order for `Kokkos.jl` to properly call this function, we must build a view from Julia whose type
matches exactly the complete type of `Kokkos::View<double*>`.
This requires `Kokkos.jl` to compile some member functions of the complete `Kokkos::View` type, as
well as its constructors.

To achieve this you have two options:
 - let `Kokkos.jl` configure the library and itself, guaranteeing that the options match
 - the library containing the function is already compiled, or you cannot/don't want to change its
   Kokkos configuration: you must configure `Kokkos.jl` with the exact same options

In both cases, this is how you can configure `Kokkos.jl` for:
 - the element type: must be present in [view_types](@ref)
 - the dimension: must be present in [view_dims](@ref)
 - the [`Layout`](@ref): must be present in [view_layouts](@ref)
 - the [`MemorySpace`](@ref): configured through the [backends](@ref) options
 - the memory traits: not yet implemented

All possible combinations of those parameters are compiled when loading `Kokkos.jl`.
For example, if `view_types = [Float64, Float32]`, `view_dims = [1, 2]`,
`view_layouts = ["left", "right"]` and `backends = [Serial, Cuda]`, there will be a total of
`2×2×2×2 = 16` different views compiled.

!!! note

    Your project is not affected by the [view_dims](@ref), [view_types](@ref) and
    [view_layouts](@ref) options.
    You only need to make sure that the combination of those options cover all usages of views in
    your project.


## Your CMake project

The CMake project shouldn't need extra handling:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(MyLib)

set(CMAKE_CXX_EXTENSIONS OFF)  # Kokkos will warn you if this is not set to OFF

find_package(Kokkos REQUIRED)

add_library(MyLib SHARED my_lib.cpp)
target_link_libraries(MyLib PRIVATE Kokkos::kokkos)  # or PUBLIC, it doesn't matter
```

`find_package` requires the `Kokkos_ROOT` (or `Kokkos_DIR`) variable to be set when configuring the
project. `Kokkos.jl` can do that for you.
The advantage of this approach is that your project and `Kokkos.jl` will share the same Kokkos
installation, reducing the compilation time.

If your project uses Kokkos in-tree, you have several options: 
 - keep the call to `add_subdirectory` the same, and configure [kokkos_path](@ref) to use the same
   path
 - change it to `add_subdirectory(${Kokkos_ROOT} lib/kokkos)` (the second argument is arbitrary)


## Loading the wrapper library of `Kokkos.jl`

`Kokkos.jl` relies on a wrapper library written in C++ to compile all possible combinations of
`Kokkos::View`, as described by the [Configuration Options](@ref), as well as all
[`ExecutionSpace`](@ref), [`MemorySpace`](@ref) needed and [other functions](@ref Environment).

Upon loading `Kokkos.jl`, this wrapper library is not loaded (and maybe not compiled).
Therefore, most Kokkos functions will not work (some methods might be missing, others will raise an
error).

To load the wrapper library you can use [`Kokkos.load_wrapper_lib`](@ref) or
[`Kokkos.initialize`](@ref):

```julia-repl
julia> using Kokkos

julia> Kokkos.load_wrapper_lib()  # Will compile then load the library, it may take some time

julia> Kokkos.initialize()  # Will also call `Kokkos.load_wrapper_lib()` if needed

```

!!! note
    The reason the wrapper library is not loaded when `using Kokkos`, is for setting the
    [Configuration Options](@ref).
    After `Kokkos.load_wrapper_lib()` has been called, the configuration options are locked, and
    require to restart the Julia session for changes to be applied.

!!! note
    Setting the environment variable `JULIA_DEBUG` to `Kokkos` will print all steps and commands
    called to compile and load the wrapper library, as well as for user libraries.


## Compiling and loading the library

By default, when loading `Kokkos.jl` the build files will be stored in "./.kokkos-build/", this can
be configured with [build_dir](@ref).
It is recommended to build the project files to same directory, by using the
`Kokkos.KOKKOS_BUILD_DIR` variable.
In order for the [Configuration Options](@ref) to be passed correctly, you should use a
`CMakeKokkosProject`:

```julia-repl
julia> my_lib_path = "./path/to/mylib/project"
"./path/to/mylib/project"

julia> my_lib_build_path = joinpath(Kokkos.KOKKOS_BUILD_DIR, "mylib")
"./.kokkos-build/mylib"

julia> project = CMakeKokkosProject(my_lib_path, "libMyLib";
                                    target="MyLib", build_dir=my_lib_build_path)
Kokkos project from sources located at './path/to/mylib/project'
Building in './.kokkos-build/mylib'
...
```

If `target` is not given, `CMakeKokkosProject` will build by default all targets of the CMake
project.
Here `"libMyLib"` is the name of the result of the `MyLib` target: the library we want to compile
and load.

```julia-repl
julia> compile(project)

julia> my_lib = load_lib(project)
CLibrary(...)
```

The library can then be used the same way as you would with a shared library.
Use [`handle`](@ref) to get a pointer to pass to `Libdl.dlsym` or use [`get_symbol`](@ref) to get
the address of our `fill_view` function:

```julia-repl
julia> v = Kokkos.View{Float64}(undef, 10)
10-element Kokkos.Views.View1D_HostAllocated{Float64}:
 6.365987373e-314
 1.14495326e-316
...

julia> ccall(get_symbol(my_lib, :fill_view),
             Cvoid, (Ref{Kokkos.View}, Float64),
             v, 0.1)

julia> v
10-element Kokkos.Views.View1D_HostAllocated{Float64}:
 0.1
 0.1
...
```

Here we called `void fill_view(Kokkos::View<double>&, double)`, which has been compiled with a
_single_ set of template arguments for `Kokkos::View`. Therefore the `ccall` is only valid if the
view passed to it matches exactly those template arguments. You can further specify the argument
types of the `ccall` to reflect this:

```julia-repl
julia> ccall(get_symbol(my_lib, :fill_view),
             Cvoid, (Ref{Kokkos.View{Float64, 1, Kokkos.DEFAULT_DEVICE_MEM_SPACE}}, Float64),
             v, 0.1)
```

The library is opened in a way which allows it to be unloaded afterward using [`unload_lib`](@ref):

```julia-repl
julia> unload_lib(my_lib)
true

julia> is_lib_loaded(my_lib)
false
```

This can be useful in order to reconfigure and recompile the project in the same session, to perform
compilation parameters exploration for example.
As long as all views are allocated through `Kokkos.jl`, they can be safely re-used after a library
reload.

!!! warning

    If any view which has been allocated by an external library is owned by Julia, it *must* be
    finalized before unloading the library (i.e. either
    [`finalize`](https://docs.julialang.org/en/v1/base/base/#Base.finalize) must be called on the
    view, or the garbage collector did so automatically beforehand).

    Failure to do so will result in nasty segfaults when the GC tries to call the finalizer on the
    view, which also happens when Julia is exiting.

    The segfault could look like this:
    ```
    signal (11): Segmentation error
    in expression starting at /home/Kokkos/test/runtests.jl:19
    unknown function (ip: 0x7f928f488090)
    _ZN6Kokkos4Impl22SharedAllocationRecordIvvE9decrementEPS2_ at /home/Kokkos/.kokkos-build/wrapper-build-release/lib/kokkos/core/src/libkokkoscore.so.4.0 (unknown line)
    ```
    The main clue that it is a finalizer error is the fact it happens in
    `Kokkos::Impl::SharedAllocationRecord::decrement`.
