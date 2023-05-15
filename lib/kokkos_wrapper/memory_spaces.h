
#ifndef KOKKOS_WRAPPER_MEMORY_SPACES_H
#define KOKKOS_WRAPPER_MEMORY_SPACES_H

#include "spaces.h"
#include "utils.h"

#include "Kokkos_Core.hpp"


template<>
struct SpaceInfo<Kokkos::HostSpace>
{
    using space = Kokkos::HostSpace;
    static constexpr const char* julia_name = "HostSpace";
};

#ifdef KOKKOS_ENABLE_CUDA
template<>
struct SpaceInfo<Kokkos::CudaSpace>
{
    using space = Kokkos::CudaSpace;
    static constexpr const char* julia_name = "CudaSpace";
};

template<>
struct SpaceInfo<Kokkos::CudaHostPinnedSpace>
{
    using space = Kokkos::CudaHostPinnedSpace;
    static constexpr const char* julia_name = "CudaHostPinnedSpace";
};

#if (KOKKOS_VERSION >= 40000) || defined(KOKKOS_ENABLE_CUDA_UVM)
// KOKKOS_ENABLE_CUDA_UVM is implicitly ON after version 4
template<>
struct SpaceInfo<Kokkos::CudaUVMSpace>
{
    using space = Kokkos::CudaUVMSpace;
    static constexpr const char* julia_name = "CudaUVMSpace";
};
#endif // (KOKKOS_VERSION >= 40000) || defined(KOKKOS_ENABLE_CUDA_UVM)
#endif // KOKKOS_ENABLE_CUDA

#if KOKKOS_ENABLE_HIP
#if KOKKOS_VERSION_GREATER_EQUAL(4, 0, 0)
namespace Kokkos_HIP = Kokkos;
#else
namespace Kokkos_HIP = Kokkos::Experimental;
#endif

template<>
struct SpaceInfo<Kokkos_HIP::HIPSpace>
{
    using space = Kokkos_HIP::HIPSpace;
    static constexpr const char* julia_name = "HIPSpace";
};

template<>
struct SpaceInfo<Kokkos_HIP::HIPHostPinnedSpace>
{
    using space = Kokkos_HIP::HIPHostPinnedSpace;
    static constexpr const char* julia_name = "HIPHostPinnedSpace";
};

template<>
struct SpaceInfo<Kokkos_HIP::HIPManagedSpace>
{
    using space = Kokkos_HIP::HIPManagedSpace;
    static constexpr const char* julia_name = "HIPManagedSpace";
};
#endif // KOKKOS_ENABLE_HIP


/**
 * Template list of all enabled Kokkos memory spaces
 */
using MemorySpacesList = TList<
          Kokkos::HostSpace

#ifdef KOKKOS_ENABLE_CUDA
        , Kokkos::CudaSpace
        , Kokkos::CudaHostPinnedSpace
#if (KOKKOS_VERSION >= 40000) || defined(KOKKOS_ENABLE_CUDA_UVM)
        , Kokkos::CudaUVMSpace
#endif // (KOKKOS_VERSION >= 40000) || defined(KOKKOS_ENABLE_CUDA_UVM)
#endif // KOKKOS_ENABLE_CUDA

#if KOKKOS_ENABLE_HIP
        , Kokkos_HIP::HIPSpace
        , Kokkos_HIP::HIPHostPinnedSpace
        , Kokkos_HIP::HIPManagedSpace
#endif // KOKKOS_ENABLE_HIP

#if KOKKOS_ENABLE_SYCL
#error "SYCL memory spaces are not yet supported"
//        , Kokkos::Experimental::SYCLDeviceUSMSpace
//        , Kokkos::Experimental::SYCLSharedUSMSpace
//        , Kokkos::Experimental::SYCLHostUSMSpace
#endif // KOKKOS_ENABLE_SYCL
>;


#endif //KOKKOS_WRAPPER_MEMORY_SPACES_H
