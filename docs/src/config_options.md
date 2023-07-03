```@meta
CurrentModule = Kokkos
```

# Configuration Options

Configuration options are set using [`Preferences.jl`](https://github.com/JuliaPackaging/Preferences.jl).
Your `LocalPreferences.jl` file will store the options needed by your current project in a
`[Kokkos]` section.

!!! danger "Important"

    Unlike some packages using `Preferences.jl`, it is possible to change all options during the
    same Julia session, using their setters.
    However, if you need to dynamically configure `Kokkos.jl`, it must be done before loading the
    wrapper library.
    After calling [`load_wrapper_lib`](@ref) (or [`initialize`](@ref)), all options will be
    locked, and any changes made afterward will **not** affect the current Julia session.

### kokkos_version

The version of Kokkos to use. Must be a valid version tag in the official Kokkos repository (e.g.
`"4.0.00"`).

Only used when [kokkos_path](@ref) is not set, and defaults to the one of the packaged sources of
kokkos.

Each version is stored in the package's scratch space, which is checked-out upon loading the Kokkos.

Can be set using `Kokkos.set_kokkos_version()`.
The value for the current Julia session is stored in `Kokkos.LOCAL_KOKKOS_VERSION_STR`.

### kokkos_path

The path to the Kokkos sources (not an installation!) to use.
If not set, it defaults to the Kokkos version packaged with `Kokkos.jl`.

Can be set using `Kokkos.set_kokkos_path()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_PATH`.

### cmake_options

The list of CMake options to pass to all [`CMakeKokkosProjects`](@ref CMakeKokkosProject).

Can be set using `Kokkos.set_cmake_options()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_CMAKE_OPTIONS`.

### kokkos_options

The list of Kokkos options to pass to all [`KokkosProjects`](@ref KokkosProject).

It can be passed as a list of `"Kokkos_<option_name>=<value>"`, or as a `Dict{String, Any}` (`Bool`
values will be converted to `"ON"` and `"OFF"`, others to strings).

Can be set using `Kokkos.set_kokkos_options()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_LIB_OPTIONS`.

### backends

The list of Kokkos backends to compile for. When in uppercase and prefixed by `Kokkos_ENABLE_` the
names should correspond to one of the valid
[device backends options](https://kokkos.github.io/kokkos-core-wiki/keywords.html#device-backends).
Defaults to `Serial` and `OpenMP`.

Can be set using `Kokkos.set_backends()`, using a vector of `String` or [`ExecutionSpace`](@ref)
subtypes.

The value for the current Julia session is stored in `Kokkos.KOKKOS_BACKENDS`.

### build_type

CMake build type.

Can be set using `Kokkos.set_build_type()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_BUILD_TYPE`.

### build_dir

Main building directory for the current session.
The wrapping library is built in `$(build_dir)/wrapper-build-$(build_type)/`.

Can be set using `Kokkos.set_build_dir()`, [`Kokkos.build_in_scratch`](@ref build_in_scratch),
[`Kokkos.build_in_tmp`](@ref build_in_tmp) or [`Kokkos.build_in_project`](@ref build_in_project).
The value for the current Julia session is stored in `Kokkos.KOKKOS_BUILD_DIR`.

!!! note

    `Kokkos.set_build_dir()` accepts a kwarg `local_only` (default: `false`) which allows to set the
    build directory without writing to `LocalPreferences.toml`.
    
    This is only useful in MPI applications, where only one process should modify the confiuration
    options and compile code.
    Calling `Kokkos.set_build_dir(build_dir; local_only=true)` on non-root processes will allow them
    to find the libraries compiled by the root process.
