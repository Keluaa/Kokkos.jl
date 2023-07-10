
#ifndef KOKKOS_WRAPPER_KOKKOS_UTILS_H
#define KOKKOS_WRAPPER_KOKKOS_UTILS_H

#include "Kokkos_Core.hpp"


#define KOKKOS_VERSION_CMP(OP, MAJOR, MINOR, PATCH) \
  (KOKKOS_VERSION OP ((MAJOR)*10000 + (MINOR)*100 + (PATCH)))


#if KOKKOS_VERSION_CMP(>=, 4, 0, 0)
namespace Kokkos_HIP = Kokkos;
#else
namespace Kokkos_HIP = Kokkos::Experimental;
#endif // KOKKOS_VERSION_CMP(>=, 4, 0, 0)


#endif //KOKKOS_WRAPPER_KOKKOS_UTILS_H
