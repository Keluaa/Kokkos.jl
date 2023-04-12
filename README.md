# Kokkos.jl: A Kokkos wrapper for Julia 

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://keluaa.github.io/Kokkos.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://keluaa.github.io/Kokkos.jl/dev)

A Julia wrapper around the popular [Kokkos](https://github.com/kokkos/kokkos) C++ library.

This package allows to create `Kokkos::View` objects, use them as an `Array` in Julia, and call a C++ Kokkos library using `ccall`.
Most basic functionnalities of Kokkos (initialization, execution and memory spaces) are available.

If the library you want to use is configured with CMake, it is possible to configure the project with `Kokkos.jl`.

This package uses a wrapper library which is compiled when loading the package.
Because it is not pre-compiled in an artifact, it maximizes the flexibility of usage of `Kokkos.jl`.

`Kokkos.jl` currently supports Kokkos v3.7.0, v3.7.1, v4.0.0 and above.
All Kokkos backends should be supported by this package, but not all of them were tested.
