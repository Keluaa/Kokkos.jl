# Kokkos.jl: A Kokkos wrapper for Julia 

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://keluaa.github.io/Kokkos.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://keluaa.github.io/Kokkos.jl/dev)
[![Build Status](https://github.com/Keluaa/Kokkos.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Keluaa/Kokkos.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Keluaa/Kokkos.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Keluaa/Kokkos.jl)

A Julia wrapper around the popular [Kokkos](https://github.com/kokkos/kokkos) C++ library, using [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl).

This package allows to create `Kokkos::View` objects, use them as an `Array` in Julia, and call a C++ Kokkos library through `ccall` with those views.
Most basic functionnalities of Kokkos (initialization, views, fences, memory and execution spaces) are available.

If the library you want to use is configured with CMake, it is possible to configure the project with `Kokkos.jl`.

This package uses a wrapper library which is compiled before initializing Kokkos.
Because it is not pre-compiled as an artifact, this maximizes the flexibility of usage of `Kokkos.jl`.

`Kokkos.jl` currently supports Kokkos v3.7.0, v3.7.1, v4.0.0 and above.
All Kokkos backends should be supported by this package, but not all of them were tested.

### Supported functionalities
 * :heavy_check_mark: `Kokkos::initialize`, `Kokkos::finalize`, `Kokkos::InitializationSettings`
 * :heavy_check_mark: `Kokkos::View`, `Kokkos::View<T, SomeMemorySpace>`, `Kokkos::view_alloc`
 * :x: `Kokkos::View<T, MyLayout, MyMemoryTraits>` (planned)
 * :heavy_check_mark: `Kokkos::create_mirror`, `Kokkos::create_mirror_view`
 * :heavy_check_mark: `Kokkos::deep_copy`
 * :x: `Kokkos::resize`, `Kokkos::realloc` (planned)
 * :x: `Kokkos::subview` (planned)
 * :x: All parallel patterns (`Kokkos::parallel_for`, `Kokkos::parallel_reduce`, `Kokkos::parallel_scan`), reducers, execution policies and tasking
 * :heavy_check_mark: `Kokkos::fence`
 * :heavy_check_mark: All execution spaces (`Kokkos::OpenMP`, `Kokkos::Cuda`...) and memory spaces (`Kokkos::HostSpace`, `Kokkos::CudaSpace`...)
 * :x: Atomics
 * :x: All containers (`Kokkos::DualView`, `Kokkos::ScatterView`...) (planned)
 * :x: SIMD
 
### Tested backends
 * :heavy_check_mark: `Kokkos::Serial`
 * :heavy_check_mark: `Kokkos::OpenMP`
 * :x: `Kokkos::Threads`
 * :x: `Kokkos::HPX`
 * :x: `Kokkos::OpenMPTarget`
 * :x: `Kokkos::Cuda`
 * :x: `Kokkos::HIP`
 * :x: `Kokkos::SYCL`
 * :x: `Kokkos::OpenACC`
