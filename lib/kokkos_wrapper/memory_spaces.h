
#ifndef KOKKOS_WRAPPER_MEMORY_SPACES_H
#define KOKKOS_WRAPPER_MEMORY_SPACES_H

#include "Kokkos_Core.hpp"

#include "spaces.h"
#include "utils.h"
#include "kokkos_utils.h"
#include "parameters.h"


template<>
struct SpaceInfo<Kokkos::HostSpace>
{
    using space = Kokkos::HostSpace;
    static constexpr std::string_view julia_name = "HostSpace";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::HostSpace> : std::false_type {};


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

template<>
struct jlcxx::IsMirroredType<Kokkos::CudaSpace> : std::false_type {};
template<>
struct jlcxx::IsMirroredType<Kokkos::CudaHostPinnedSpace> : std::false_type {};

#if KOKKOS_VERSION_CMP(>=, 4, 0, 0) || defined(KOKKOS_ENABLE_CUDA_UVM)
// KOKKOS_ENABLE_CUDA_UVM is implicitly ON after version 4
template<>
struct SpaceInfo<Kokkos::CudaUVMSpace>
{
    using space = Kokkos::CudaUVMSpace;
    static constexpr std::string_view julia_name = "CudaUVMSpace";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::CudaUVMSpace> : std::false_type {};
#endif // KOKKOS_VERSION_CMP(>=, 4, 0, 0) || defined(KOKKOS_ENABLE_CUDA_UVM)
#endif // KOKKOS_ENABLE_CUDA

#ifdef KOKKOS_ENABLE_HIP
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

template<>
struct jlcxx::IsMirroredType<Kokkos_HIP::HIPSpace> : std::false_type {};
template<>
struct jlcxx::IsMirroredType<Kokkos_HIP::HIPHostPinnedSpace> : std::false_type {};
template<>
struct jlcxx::IsMirroredType<Kokkos_HIP::HIPManagedSpace> : std::false_type {};
#endif // KOKKOS_ENABLE_HIP


/**
 * Template list of all enabled Kokkos memory spaces
 */
using MemorySpacesList = TList<
          Kokkos::HostSpace

#ifdef KOKKOS_ENABLE_CUDA
        , Kokkos::CudaSpace
        , Kokkos::CudaHostPinnedSpace
#if KOKKOS_VERSION_CMP(>=, 4, 0, 0) || defined(KOKKOS_ENABLE_CUDA_UVM)
        , Kokkos::CudaUVMSpace
#endif // KOKKOS_VERSION_CMP(>=, 4, 0, 0) || defined(KOKKOS_ENABLE_CUDA_UVM)
#endif // KOKKOS_ENABLE_CUDA

#ifdef KOKKOS_ENABLE_HIP
        , Kokkos_HIP::HIPSpace
        , Kokkos_HIP::HIPHostPinnedSpace
        , Kokkos_HIP::HIPManagedSpace
#endif // KOKKOS_ENABLE_HIP

#ifdef KOKKOS_ENABLE_SYCL
#error "SYCL memory spaces are not yet supported"
//        , Kokkos::Experimental::SYCLDeviceUSMSpace
//        , Kokkos::Experimental::SYCLSharedUSMSpace
//        , Kokkos::Experimental::SYCLHostUSMSpace
#endif // KOKKOS_ENABLE_SYCL
>;


static constexpr std::string_view MEM_SPACE_STR = AS_STR(MEM_SPACE);
static constexpr std::string_view DEST_MEM_SPACE_STR = AS_STR(DEST_MEM_SPACE);

// The memory space with the same name as `MEM_SPACE`, or `void`
using MemorySpace = decltype(find_space<MEM_SPACE_STR, void>(MemorySpacesList{}))::Arg<0>;

// The memory space with the same name as `DEST_MEM_SPACE`, or `void`
using DestMemorySpace = decltype(find_space<DEST_MEM_SPACE_STR, void>(MemorySpacesList{}))::Arg<0>;

#endif //KOKKOS_WRAPPER_MEMORY_SPACES_H
