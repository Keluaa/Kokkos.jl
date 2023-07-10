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

* :heavy_check_mark: `Kokkos::initialize`, `Kokkos::finalize` and `Kokkos::InitializationSettings`
* :heavy_check_mark: `Kokkos::View`, `Kokkos::View<T, MyLayout, SomeMemorySpace>` and `Kokkos::view_alloc`
* :x: `Kokkos::MemoryTraits` (planned)
* :heavy_check_mark: `Kokkos::create_mirror`, `Kokkos::create_mirror_view`
* :heavy_check_mark: `Kokkos::deep_copy`
* :heavy_check_mark: `Kokkos::subview`
* :x: `Kokkos::resize`, `Kokkos::realloc` (planned)
* :heavy_check_mark: `Kokkos::fence`
* :heavy_check_mark: All execution spaces (`Kokkos::OpenMP`, `Kokkos::Cuda`...) and memory spaces (`Kokkos::HostSpace`, `Kokkos::CudaSpace`...)
* :x: All parallel patterns (`Kokkos::parallel_for`, `Kokkos::parallel_reduce`, `Kokkos::parallel_scan`), reducers, execution policies and tasking
* :x: Atomics
* :x: All containers (`Kokkos::DualView`, `Kokkos::ScatterView`...) (planned)
* :x: SIMD
* :x: View hooks

## Tested backends

* :heavy_check_mark: `Kokkos::Serial`
* :heavy_check_mark: `Kokkos::OpenMP`
* :x: `Kokkos::Threads`
* :x: `Kokkos::HPX`
* :x: `Kokkos::OpenMPTarget`
* :heavy_check_mark: `Kokkos::Cuda` + interop with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
* :heavy_check_mark: `Kokkos::HIP` + interop with [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
* :x: `Kokkos::SYCL`
* :x: `Kokkos::OpenACC`

### Known issues

* The NVCC compiler is unable to compile the wrapper library. Use Clang instead (Clang-11 is the
version used to test this backend).
* `Kokkos::Cuda ERROR: Failed to call Kokkos::Cuda::finalize()` message when exiting Julia:
`Kokkos.finalize()` is not yet called automatically upon quitting Julia, as it would require to call
the finalizers of all views before doing so.
* Memory leaks on GPU: this is a side effect of Julia's GC which cannot manage device memory. From
Julia's POV, a `Kokkos.View` is only a pointer in the host memory. Calling `GC.gc(true)` will fix the
issue.
