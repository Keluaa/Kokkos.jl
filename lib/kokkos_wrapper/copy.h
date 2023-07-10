
#ifndef KOKKOS_WRAPPER_COPY_H
#define KOKKOS_WRAPPER_COPY_H

#include "kokkos_wrapper.h"


#if defined(WRAPPER_BUILD) && COMPLETE_BUILD == 1
void define_kokkos_deep_copy(jlcxx::Module& mod);
#else
void define_kokkos_deep_copy(jlcxx::Module&) {}
#endif

#endif //KOKKOS_WRAPPER_COPY_H
