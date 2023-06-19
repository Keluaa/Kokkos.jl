#ifndef KOKKOS_WRAPPER_SUBVIEWS_H
#define KOKKOS_WRAPPER_SUBVIEWS_H

#include "views.h"


#if defined(WRAPPER_BUILD) && COMPLETE_BUILD == 1
void define_kokkos_subview(jlcxx::Module& mod);
#else
void define_kokkos_subview(jlcxx::Module&) {}
#endif

#endif //KOKKOS_WRAPPER_SUBVIEWS_H
