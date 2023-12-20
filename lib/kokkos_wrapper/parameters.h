
#ifndef KOKKOS_WRAPPER_PARAMETERS_H
#define KOKKOS_WRAPPER_PARAMETERS_H

#ifdef WRAPPER_BUILD
// When building the wrapper library, we should not rely on dynamic build-time parameters.
// TODO: move all dependant values to some other header, and remove this to make sure no code in the wrapper lib uses any of those
#define VIEW_LAYOUT void
#define VIEW_DIMENSION
#define VIEW_TYPE
#define EXEC_SPACE_FILTER
#define MEM_SPACE_FILTER
#define DEST_LAYOUT void
#define WITHOUT_EXEC_SPACE_ARG
#define DEST_MEM_SPACES
#define WITH_NOTHING_ARG
#define SUBVIEW_DIM

#else
#include "build_parameters.h"  // Header generated at build time by 'build_parameters.sh'

// Default values to work with an IDE:
//  - view: Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultMemorySpace>
//  - deep_copy destination: on HostSpace
//  - mirror memory space: HostSpace
//  - subview dimension: 1

#ifndef VIEW_LAYOUT
#define VIEW_LAYOUT left
#endif

#ifndef VIEW_DIMENSION
#define VIEW_DIMENSION 2
#endif

#ifndef VIEW_TYPE
#define VIEW_TYPE double
#endif

#ifndef EXEC_SPACE_FILTER
#define EXEC_SPACE_FILTER
#endif

#ifndef MEM_SPACE_FILTER
#define MEM_SPACE_FILTER
#endif

#ifndef DEST_LAYOUT
#define DEST_LAYOUT VIEW_LAYOUT
#endif

#ifndef WITHOUT_EXEC_SPACE_ARG
#define WITHOUT_EXEC_SPACE_ARG 0
#endif

#ifndef DEST_MEM_SPACES
#define DEST_MEM_SPACES "HostSpace"
#endif

#ifndef WITH_NOTHING_ARG
#define WITH_NOTHING_ARG 0
#endif

#ifndef SUBVIEW_DIM
#define SUBVIEW_DIM 1
#endif

#endif //WRAPPER_BUILD


inline const char* get_params_string()
{
#define AS_STR_IMPL(x) #x
#define AS_STR(x) AS_STR_IMPL(x)
    static const char params_str[] =
          "VIEW_LAYOUT       = " AS_STR(VIEW_LAYOUT)
        "\nVIEW_DIMENSION    = " AS_STR(VIEW_DIMENSION)
        "\nVIEW_TYPE         = " AS_STR(VIEW_TYPE)
        "\nEXEC_SPACE_FILTER = " AS_STR(EXEC_SPACE_FILTER)
        "\nMEM_SPACE_FILTER  = " AS_STR(MEM_SPACE_FILTER)
        "\nDEST_LAYOUT       = " AS_STR(DEST_LAYOUT)
        "\nWITHOUT_EXEC_SPACE_ARG = " AS_STR(WITHOUT_EXEC_SPACE_ARG)
        "\nDEST_MEM_SPACES   = " AS_STR(DEST_MEM_SPACES)
        "\nWITH_NOTHING_ARG  = " AS_STR(WITH_NOTHING_ARG)
        "\nSUBVIEW_DIM       = " AS_STR(SUBVIEW_DIM);
    return params_str;
#undef AS_STR_IMPL
#undef AS_STR
}

#endif //KOKKOS_WRAPPER_PARAMETERS_H
