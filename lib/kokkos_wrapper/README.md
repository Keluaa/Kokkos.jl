
# Kokkos.jl C++ wrapper library 

The wrapper library aims to wrap most Kokkos functions with CxxWrap.jl to make them
available in Julia.

Kokkos is a C++ which uses templates everywhere, which would lead to immense
compilation times if we were to compile all possible instantiation of every type and function.
Therefore, we must restrain what to instantiate to what would be useful to the user.

The main wrapper library serves to wrap all enabled spaces, layouts, and basic Kokkos
functions (`Kokkos::initialize`, `Kokkos::fence`, etc...).

Template-heavy features are put in separate libraries, with separate independent CMake targets:
 - `views`: `Kokkos::View` methods, `Kokkos::view_alloc`, `Kokkos::view_wrap`
 - `copy`: `Kokkos::deep_copy`
 - `subviews`: `Kokkos::subview` (for all parameter combinations...)
 - `mirrors`: `Kokkos::create_mirror`, `Kokkos::create_mirror_view`

Those libraries are meant to be compiled and loaded at any time, when needed by the user.
The types and functions covered are restrained by macros defined at build-time
(NOT at configuration time!) through the `build_parameters.h` file generated from environment
variables before compilation:

 - `VIEW_DIMENSIONS`: comma-separated list of dimensions to instantiate 
 - `VIEW_TYPES`: comma-separated list of C++ types to instantiate
 - `VIEW_LAYOUTS`: comma-separated list of layouts to instantiate, allows some aliases:
    - `left`, `right`, `stride` are aliases for
      `Kokkos::LayoutLeft`, `Kokkos::LayoutRight` and `Kokkos::LayoutStride` respectively
    - `deviceDefault` is equivalent to `Kokkos::DefaultExecutionSpace::array_layout`
    - `hostDefault` is equivalent to `Kokkos::DefaultHostExecutionSpace::array_layout`
 - `EXEC_SPACE_FILTER`: comma-separated list of names of execution spaces (e.g. `"Host", "Cuda""`)
   to instantiate. If empty, defaults to all enabled Kokkos execution spaces
 - `MEM_SPACE_FILTER`: comma-separated list of names of memory spaces (e.g. `"HostSpace", "CudaSpace"`)
   to instantiate. If empty, default to all enabled Kokkos memory spaces

Some variables are specific to some functions:
 - `Kokkos::deep_copy`
   - `DEST_LAYOUTS`: same as `VIEW_LAYOUTS` for destination view layouts.
     Defaults to `VIEW_LAYOUTS`.
   - `DEST_MEM_SPACES`: comma-separated list of names of destination memory spaces
     to instantiate. Behaves similarly to `MEM_SPACE_FILTER`, but if empty defaults to an empty list.
   - `WITHOUT_EXEC_SPACE_ARG`: bool (as an integer: `0` or `1`), whether to compile
     the version with a leading execution space parameter
 - `Kokkos::create_mirror[_view]`
   - `DEST_MEM_SPACES`: same as for `Kokkos::deep_copy` 
   - `WITH_NOTHING_ARG`: bool (as an integer: `0` or `1`), whether to compile the
     version with an implicit memory space destination, which defaults to a host-accessible
     memory space
 - `Kokkos::subview`
   - `SUBVIEW_DIMS`: comma-separated list of target dimensions of subviews.
     Defaults to `VIEW_DIMENSIONS`.

While debugging, you can use functions in `printing_utils.h` to print any type or any `TList`
with no type-mangling.
