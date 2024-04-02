# Kokkos.jl: A Kokkos wrapper for Julia

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://keluaa.github.io/Kokkos.jl/stable)
[![Dev documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://keluaa.github.io/Kokkos.jl/dev)
[![Build Status](https://github.com/Keluaa/Kokkos.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Keluaa/Kokkos.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Keluaa/Kokkos.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Keluaa/Kokkos.jl)

A Julia wrapper around the popular [Kokkos](https://github.com/kokkos/kokkos) C++ library, using
[CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl).

This package allows to create `Kokkos::View` objects, use them as an `Array` in Julia, and call a
C++ Kokkos library through `ccall` with those views.
Most basic functionalities of Kokkos (initialization, views, subviews, copies, fences, memory and
execution spaces) are available.
With [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) it is possible to use any view with MPI
seamlessly.

`Kokkos.jl` does not currently offer the possibility to code Kokkos kernels in Julia, they must be
written in a separate C++ shared library.
If the library you want to use is configured with CMake, it is possible to configure the project
with `Kokkos.jl`.

This package relies on a wrapper library which is compiled when initializing Kokkos, which is
configured with the CMake and Kokkos options set in the configuration options.
Because it is not pre-compiled as an artifact, this maximizes the flexibility of usage of `Kokkos.jl`.
However, most Kokkos functions (and views) are compiled separately on demand, when their respective
Julia method is called for the first time.
The resulting shared library is then cached for the next session.

`Kokkos.jl` currently supports Kokkos v3.7, v4.0 and above.
All Kokkos backends should be supported by this package, but not all of them were tested (yet).

## Supported functionalities

* :white_check_mark: `Kokkos::initialize`, `Kokkos::finalize` and `Kokkos::InitializationSettings`
* :white_check_mark: `Kokkos::View`, `Kokkos::View<T, MyLayout, SomeMemorySpace>` and `Kokkos::view_alloc`
* :x: `Kokkos::MemoryTraits` (planned)
* :white_check_mark: `Kokkos::create_mirror`, `Kokkos::create_mirror_view`
* :white_check_mark: `Kokkos::deep_copy`
* :white_check_mark: `Kokkos::subview`
* :x: `Kokkos::resize`, `Kokkos::realloc` (planned)
* :white_check_mark: `Kokkos::fence`
* :white_check_mark: All execution spaces (`Kokkos::OpenMP`, `Kokkos::Cuda`...) and memory spaces (`Kokkos::HostSpace`, `Kokkos::CudaSpace`...)
* :x: All parallel patterns (`Kokkos::parallel_for`, `Kokkos::parallel_reduce`, `Kokkos::parallel_scan`), reducers, execution policies and tasking
* :x: Atomics
* :x: All containers (`Kokkos::DualView`, `Kokkos::ScatterView`...) (planned)
* :x: SIMD
* :x: View hooks

## Supported backends

* :white_check_mark: `Kokkos::Serial`
* :white_check_mark: `Kokkos::OpenMP`
* :white_check_mark: `Kokkos::Cuda`* + interop with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (tested with Clang and nvcc)
* :white_check_mark: `Kokkos::HIP`* + interop with [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
* :white_check_mark: `Kokkos::Threads`*
* :white_check_mark: `Kokkos::HPX`*
* :question: `Kokkos::OpenMPTarget`**
* :question: `Kokkos::OpenACC`**
* :x: `Kokkos::SYCL`* (interop with [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) is planned)

\*: tested locally, not through GitHub CI
<br>**: can compile, not tested as the is backend too experimental

## Known issues

* Memory leaks on GPU: this is a side effect of Julia's GC which cannot manage device memory. From
Julia's POV, a `Kokkos.View` is only a pointer in the host memory. Calling `GC.gc(true)` manually
will fix the issue.
* The NVHPC compiler cannot correctly compile the wrapper library due to compiler bugs. There is currently no plan to fix this issue.
