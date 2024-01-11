
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

 - `VIEW_DIMENSION`: dimension to instantiate 
 - `VIEW_TYPE`: C++ type to instantiate
 - `VIEW_LAYOUT`: layout to instantiate, allows some aliases:
    - `left`, `right`, `stride` are aliases for
      `Kokkos::LayoutLeft`, `Kokkos::LayoutRight` and `Kokkos::LayoutStride` respectively
    - `deviceDefault` is equivalent to `Kokkos::DefaultExecutionSpace::array_layout`
    - `hostDefault` is equivalent to `Kokkos::DefaultHostExecutionSpace::array_layout`
    - `NONE` is for `void`
 - `EXEC_SPACE`: name of execution space (e.g. `"Host", "Cuda"`) to instantiate. Defaults to `void`.
 - `MEM_SPACE`: name of memory space (e.g. `"HostSpace", "CudaSpace"`) to instantiate. Defaults to `void`.

Some variables are specific to some functions:
 - `Kokkos::deep_copy`
   - `DEST_LAYOUT`: same as `VIEW_LAYOUT` for the destination view layout. Defaults to `VIEW_LAYOUT`.
   - `DEST_MEM_SPACE`: same as `MEM_SPACE` for the destination memory space. Defaults to `void`.
   - `WITHOUT_EXEC_SPACE_ARG`: bool (as an integer: `0` or `1`), whether to compile
     the version with a leading execution space parameter.
 - `Kokkos::create_mirror[_view]`
   - `DEST_MEM_SPACE`: same as for `Kokkos::deep_copy` 
   - `WITH_NOTHING_ARG`: bool (as an integer: `0` or `1`), whether to compile the
     version with an implicit memory space destination, which defaults to a host-accessible
     memory space.
 - `Kokkos::subview`
   - `SUBVIEW_DIM`: target dimension of the subview to instantiate.

While debugging, you can use functions in `printing_utils.h` to print any type or any `TList`
with no type-mangling.
