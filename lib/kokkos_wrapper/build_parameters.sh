#!/usr/bin/env bash

# shellcheck disable=SC2153

# Remove quotes, as they should not appear in macros
p_VIEW_LAYOUT=$(echo "$VIEW_LAYOUT" | tr -d '"')
p_VIEW_DIM=$(echo "$VIEW_DIM" | tr -d '"')
p_VIEW_TYPE=$(echo "$VIEW_TYPE" | tr -d '"')
p_EXEC_SPACES=$EXEC_SPACES
p_MEM_SPACES=$MEM_SPACES
p_DEST_LAYOUT=$DEST_LAYOUT
p_DEST_MEM_SPACES=$DEST_MEM_SPACES
p_WITHOUT_EXEC_SPACE_ARG=$(echo "$WITHOUT_EXEC_SPACE_ARG" | tr -d '"')
p_WITH_NOTHING_ARG=$(echo "$WITH_NOTHING_ARG" | tr -d '"')
p_SUBVIEW_DIM=$(echo "$SUBVIEW_DIM" | tr -d '"')


# build_parameters.h ends up in the current build directory
cat > ./build_parameters.h <<- END_OF_FILE

#ifndef KOKKOS_WRAPPER_BUILD_PARAMETERS_H
#define KOKKOS_WRAPPER_BUILD_PARAMETERS_H

// common parameters
#define VIEW_LAYOUT $p_VIEW_LAYOUT
#define VIEW_DIMENSION $p_VIEW_DIM
#define VIEW_TYPE $p_VIEW_TYPE
#define EXEC_SPACE_FILTER $p_EXEC_SPACES
#define MEM_SPACE_FILTER $p_MEM_SPACES

// copy.cpp parameters
#define DEST_LAYOUT $p_DEST_LAYOUT
#define WITHOUT_EXEC_SPACE_ARG $p_WITHOUT_EXEC_SPACE_ARG

// mirrors.cpp parameters
#define DEST_MEM_SPACES $p_DEST_MEM_SPACES
#define WITH_NOTHING_ARG $p_WITH_NOTHING_ARG

// subviews.cpp parameters
#define SUBVIEW_DIM $p_SUBVIEW_DIM

#endif // KOKKOS_WRAPPER_BUILD_PARAMETERS_H

END_OF_FILE
