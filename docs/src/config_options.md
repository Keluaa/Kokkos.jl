```@meta
CurrentModule = Kokkos
```

# Configuration Options

Configuration options are set using [`Preferences.jl`](https://github.com/JuliaPackaging/Preferences.jl).
Your `LocalPreferences.jl` file will store the options needed by your current project in a `[Kokkos]` section.


### kokkos_version

The version of Kokkos to use. Must be a valid version tag in the official Kokkos repository (e.g.
"4.0.00").

Only used when [kokkos_path](@ref) is not set, and defaults to the packaged sources of kokkos.

Each version is stored in the package's scratch space, and checked-out when loading the package.

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

Can be set using `Kokkos.set_kokkos_options()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_LIB_OPTIONS`.


### backends

The list of Kokkos backends to compile for. When in uppercase and prefixed by `Kokkos_ENABLE_` the names should
correspond to one of the valid [device backends options](https://kokkos.github.io/kokkos-core-wiki/keywords.html#device-backends).
Defaults to `Serial` and `OpenMP`.

Can be set using `Kokkos.set_backends()`, using a vector of `String` or [`ExecutionSpace`](@ref) sub-types.

The value for the current Julia session is stored in `Kokkos.KOKKOS_BACKENDS`.


### view_dims

List of `Int`s for which view dimensions will be compiled.

Can be set using `Kokkos.set_view_dims()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_VIEW_DIMS`.


### view_types

List of `Type`s for which views will be compiled.

Can be set using `Kokkos.set_view_types()`, using a `Vector` of `String` or `Type`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_VIEW_TYPES`.


### build_type

CMake build type.

Can be set using `Kokkos.set_build_type()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_BUILD_TYPE`.


### build_dir

Main building directory for the current session.
The wrapping library is built in `$(build_dir)/wrapper-build-$(build_type)/`.

Can be set using `Kokkos.set_build_dir()`.
The value for the current Julia session is stored in `Kokkos.KOKKOS_BUILD_DIR`.
