
#ifndef KOKKOS_WRAPPER_EXECUTION_SPACES_H
#define KOKKOS_WRAPPER_EXECUTION_SPACES_H

#include "spaces.h"

#include "Kokkos_Core.hpp"


#ifdef KOKKOS_ENABLE_SERIAL
template<>
struct SpaceInfo<Kokkos::Serial> {
    using space = Kokkos::Serial;
    static constexpr const char* julia_name = "Serial";
};
#endif


#ifdef KOKKOS_ENABLE_OPENMP
template<>
struct SpaceInfo<Kokkos::OpenMP> {
    using space = Kokkos::OpenMP;
    static constexpr const char* julia_name = "OpenMP";
};
#endif


#ifdef KOKKOS_ENABLE_OPENACC
template<>
struct SpaceInfo<Kokkos::OpenACC> {
    using space = Kokkos::OpenACC;
    static constexpr const char* julia_name = "OpenACC";
};
#endif


#ifdef KOKKOS_ENABLE_OPENMPTARGET
template<>
struct SpaceInfo<Kokkos::OpenMPTarget> {
    using space = Kokkos::OpenMPTarget;
    static constexpr const char* julia_name = "OpenMPTarget";
};
#endif


#ifdef KOKKOS_ENABLE_THREADS
template<>
struct SpaceInfo<Kokkos::Threads> {
    using space = Kokkos::Threads;
    static constexpr const char* julia_name = "Threads";
};
#endif


#ifdef KOKKOS_ENABLE_CUDA
template<>
struct SpaceInfo<Kokkos::Cuda> {
    using space = Kokkos::Cuda;
    static constexpr const char* julia_name = "Cuda";
};
#endif


#ifdef KOKKOS_ENABLE_HIP
template<>
struct SpaceInfo<Kokkos::HIP> {
    using space = Kokkos::HIP;
    static constexpr const char* julia_name = "HIP";
};
#endif


#ifdef KOKKOS_ENABLE_HPX
template<>
struct SpaceInfo<Kokkos::HPX> {
    using space = Kokkos::HPX;
    static constexpr const char* julia_name = "HPX";
};
#endif


#ifdef KOKKOS_ENABLE_SYCL
template<>
struct SpaceInfo<Kokkos::Experimental::SYCL> {
    using space = Kokkos::Experimental::SYCL;
    static constexpr const char* julia_name = "SYCL";
};
#endif


template<typename... ExecutionSpace>
struct ExecutionSpaces {};


template<typename, typename... ES>
using BuildExecutionSpacesList = ExecutionSpaces<ES...>;


/**
 * Template list of all enabled Kokkos execution spaces.
 */
using ExecutionSpaceList = BuildExecutionSpacesList<
          void  // Ignore the first element of the list for an easy way to handle commas in this declaration
#ifdef KOKKOS_ENABLE_SERIAL
        , Kokkos::Serial
#endif // KOKKOS_ENABLE_SERIAL

#ifdef KOKKOS_ENABLE_OPENMP
        , Kokkos::OpenMP
#endif // KOKKOS_ENABLE_OPENMP

#ifdef KOKKOS_ENABLE_OPENACC
        , Kokkos::OpenACC
#endif // KOKKOS_ENABLE_OPENACC

#ifdef KOKKOS_ENABLE_OPENMPTARGET
        , Kokkos::OpenMPTarget
#endif // KOKKOS_ENABLE_OPENMPTARGET

#ifdef KOKKOS_ENABLE_THREADS
        , Kokkos::Threads
#endif // KOKKOS_ENABLE_THREADS

#ifdef KOKKOS_ENABLE_CUDA
        , Kokkos::Cuda
#endif // KOKKOS_ENABLE_CUDA

#ifdef KOKKOS_ENABLE_HIP
#if KOKKOS_VERSION_GREATER_EQUAL(4, 0, 0)
        , Kokkos::HIP
#else
        , Kokkos::Experimental::HIP
#endif
#endif // KOKKOS_ENABLE_HIP

#ifdef KOKKOS_ENABLE_HPX
        , Kokkos::HPX
#endif // KOKKOS_ENABLE_HPX

#ifdef KOKKOS_ENABLE_SYCL
        , Kokkos::Experimental::SYCL
#endif // KOKKOS_ENABLE_SYCL
>;

#endif //KOKKOS_WRAPPER_EXECUTION_SPACES_H
