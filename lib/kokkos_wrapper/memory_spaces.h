
#ifndef KOKKOS_WRAPPER_MEMORY_SPACES_H
#define KOKKOS_WRAPPER_MEMORY_SPACES_H

#include "spaces.h"
#include "utils.h"

#include "Kokkos_Core.hpp"

#ifndef WRAPPER_BUILD
#include "parameters.h"
#endif


template<>
struct SpaceInfo<Kokkos::HostSpace>
{
    using space = Kokkos::HostSpace;
    static constexpr std::string_view julia_name = "HostSpace";
};

#ifdef KOKKOS_ENABLE_CUDA
template<>
struct SpaceInfo<Kokkos::CudaSpace>
{
    using space = Kokkos::CudaSpace;
    static constexpr std::string_view julia_name = "CudaSpace";
};

template<>
struct SpaceInfo<Kokkos::CudaHostPinnedSpace>
{
    using space = Kokkos::CudaHostPinnedSpace;
    static constexpr std::string_view julia_name = "CudaHostPinnedSpace";
};

#if (KOKKOS_VERSION >= 40000) || defined(KOKKOS_ENABLE_CUDA_UVM)
// KOKKOS_ENABLE_CUDA_UVM is implicitly ON after version 4
template<>
struct SpaceInfo<Kokkos::CudaUVMSpace>
{
    using space = Kokkos::CudaUVMSpace;
    static constexpr std::string_view julia_name = "CudaUVMSpace";
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
    static constexpr std::string_view julia_name = "HIPSpace";
};

template<>
struct SpaceInfo<Kokkos_HIP::HIPHostPinnedSpace>
{
    using space = Kokkos_HIP::HIPHostPinnedSpace;
    static constexpr std::string_view julia_name = "HIPHostPinnedSpace";
};

template<>
struct SpaceInfo<Kokkos_HIP::HIPManagedSpace>
{
    using space = Kokkos_HIP::HIPManagedSpace;
    static constexpr std::string_view julia_name = "HIPManagedSpace";
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


#ifndef MEM_SPACE_FILTER
#define MEM_SPACE_FILTER
#endif

#ifndef DEST_MEM_SPACES
#define DEST_MEM_SPACES
#endif


constexpr const std::array mem_space_filters = as_array<const char*>(MEM_SPACE_FILTER);
using FilteredMemorySpaceList = decltype(filter_spaces<mem_space_filters.size(), mem_space_filters>(MemorySpacesList{}));

constexpr const std::array dst_mem_spaces = as_array<const char*>(DEST_MEM_SPACES);
using DestMemSpaces = std::conditional_t<
        dst_mem_spaces.empty(),
        TList<>,
        decltype(filter_spaces<dst_mem_spaces.size(), dst_mem_spaces>(MemorySpacesList{}))
>;

static_assert(mem_space_filters.empty() || mem_space_filters.size() == FilteredMemorySpaceList::size);
static_assert(dst_mem_spaces.size() == DestMemSpaces::size);


#endif //KOKKOS_WRAPPER_MEMORY_SPACES_H
