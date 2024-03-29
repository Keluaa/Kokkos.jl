
#ifndef KOKKOS_WRAPPER_EXECUTION_SPACES_H
#define KOKKOS_WRAPPER_EXECUTION_SPACES_H

#include "Kokkos_Core.hpp"

#include "spaces.h"
#include "utils.h"
#include "kokkos_utils.h"
#include "parameters.h"


#ifdef KOKKOS_ENABLE_SERIAL
template<>
struct SpaceInfo<Kokkos::Serial> {
    using space = Kokkos::Serial;
    static constexpr std::string_view julia_name = "Serial";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::Serial> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_OPENMP
template<>
struct SpaceInfo<Kokkos::OpenMP> {
    using space = Kokkos::OpenMP;
    static constexpr std::string_view julia_name = "OpenMP";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::OpenMP> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_OPENACC
template<>
struct SpaceInfo<Kokkos::OpenACC> {
    using space = Kokkos::OpenACC;
    static constexpr std::string_view julia_name = "OpenACC";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::OpenACC> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_OPENMPTARGET
template<>
struct SpaceInfo<Kokkos::OpenMPTarget> {
    using space = Kokkos::OpenMPTarget;
    static constexpr std::string_view julia_name = "OpenMPTarget";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::OpenMPTarget> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_THREADS
template<>
struct SpaceInfo<Kokkos::Threads> {
    using space = Kokkos::Threads;
    static constexpr std::string_view julia_name = "Threads";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::Threads> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_CUDA
template<>
struct SpaceInfo<Kokkos::Cuda> {
    using space = Kokkos::Cuda;
    static constexpr std::string_view julia_name = "Cuda";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::Cuda> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_HIP
template<>
struct SpaceInfo<Kokkos_HIP::HIP> {
    using space = Kokkos_HIP::HIP;
    static constexpr std::string_view julia_name = "HIP";
};

template<>
struct jlcxx::IsMirroredType<Kokkos_HIP::HIP> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_HPX
template<>
struct SpaceInfo<Kokkos::HPX> {
    using space = Kokkos::HPX;
    static constexpr std::string_view julia_name = "HPX";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::HPX> : std::false_type {};
#endif


#ifdef KOKKOS_ENABLE_SYCL
template<>
struct SpaceInfo<Kokkos::Experimental::SYCL> {
    using space = Kokkos::Experimental::SYCL;
    static constexpr std::string_view julia_name = "SYCL";
};

template<>
struct jlcxx::IsMirroredType<Kokkos::Experimental::SYCL> : std::false_type {};
#endif


template<typename, typename... ES>
using BuildExecutionSpacesList = TList<ES...>;


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
        , Kokkos_HIP::HIP
#endif // KOKKOS_ENABLE_HIP

#ifdef KOKKOS_ENABLE_HPX
        , Kokkos::HPX
#endif // KOKKOS_ENABLE_HPX

#ifdef KOKKOS_ENABLE_SYCL
        , Kokkos::Experimental::SYCL
#endif // KOKKOS_ENABLE_SYCL
>;


static constexpr std::string_view EXEC_SPACE_STR = AS_STR(EXEC_SPACE);

// The execution space with the same name as `EXEC_SPACE`, or `void`
using ExecutionSpace = decltype(find_space<EXEC_SPACE_STR, void>(ExecutionSpaceList{}))::Arg<0>;

using Idx = typename Kokkos::RangePolicy<>::index_type;

#endif //KOKKOS_WRAPPER_EXECUTION_SPACES_H
