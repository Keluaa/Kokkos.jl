
#ifndef KOKKOS_WRAPPER_PARAMETERS_H
#define KOKKOS_WRAPPER_PARAMETERS_H

#include "kokkos_utils.h"

#if KOKKOS_VERSION_CMP(<, 4, 0, 0)
// Avoid macro redefinition warning in Kokkos 3
#undef EXEC_SPACE
#endif

#ifdef WRAPPER_BUILD
// When building the wrapper library, we should not rely on dynamic build-time parameters.
// TODO: move all dependant values to some other header, and remove this to make sure no code in the wrapper lib uses any of those
#define VIEW_LAYOUT NONE
#define VIEW_DIMENSION
#define VIEW_TYPE
#define EXEC_SPACE
#define MEM_SPACE
#define DEST_LAYOUT NONE
#define WITHOUT_EXEC_SPACE_ARG
#define DEST_MEM_SPACE
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

#ifndef EXEC_SPACE
#define EXEC_SPACE
#endif

#ifndef MEM_SPACE
#define MEM_SPACE "HostSpace"
#endif

#ifndef DEST_LAYOUT
#define DEST_LAYOUT VIEW_LAYOUT
#endif

#ifndef WITHOUT_EXEC_SPACE_ARG
#define WITHOUT_EXEC_SPACE_ARG 0
#endif

#ifndef DEST_MEM_SPACE
#define DEST_MEM_SPACE "HostSpace"
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
    static const char params_str[] =
          "VIEW_LAYOUT       = " AS_STR(VIEW_LAYOUT)
        "\nVIEW_DIMENSION    = " AS_STR(VIEW_DIMENSION)
        "\nVIEW_TYPE         = " AS_STR(VIEW_TYPE)
        "\nEXEC_SPACE        = " AS_STR(EXEC_SPACE)
        "\nMEM_SPACE         = " AS_STR(MEM_SPACE)
        "\nDEST_LAYOUT       = " AS_STR(DEST_LAYOUT)
        "\nWITHOUT_EXEC_SPACE_ARG = " AS_STR(WITHOUT_EXEC_SPACE_ARG)
        "\nDEST_MEM_SPACE    = " AS_STR(DEST_MEM_SPACE)
        "\nWITH_NOTHING_ARG  = " AS_STR(WITH_NOTHING_ARG)
        "\nSUBVIEW_DIM       = " AS_STR(SUBVIEW_DIM);
    return params_str;
}

#endif //KOKKOS_WRAPPER_PARAMETERS_H
