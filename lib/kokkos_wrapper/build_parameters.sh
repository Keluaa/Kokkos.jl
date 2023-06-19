
# shellcheck disable=SC2153

# Remove quotes, as they should not appear in macros
p_VIEW_LAYOUTS=$(echo "$VIEW_LAYOUTS" | tr -d '"')
p_VIEW_DIMS=$(echo "$VIEW_DIMS" | tr -d '"')
p_VIEW_TYPES=$(echo "$VIEW_TYPES" | tr -d '"')
p_EXEC_SPACES=$EXEC_SPACES
p_MEM_SPACES=$MEM_SPACES
p_DEST_LAYOUTS=$DEST_LAYOUTS
p_DEST_MEM_SPACES=$DEST_MEM_SPACES
p_WITHOUT_EXEC_SPACE_ARG=$(echo "$WITHOUT_EXEC_SPACE_ARG" | tr -d '"')
p_WITH_NOTHING_ARG=$(echo "$WITH_NOTHING_ARG" | tr -d '"')
p_SUBVIEW_DIMS=$(echo "$SUBVIEW_DIMS" | tr -d '"')


# parameters.h ends up in the current build directory
cat > ./parameters.h <<- END_OF_FILE

#ifndef KOKKOS_WRAPPER_PARAMETERS_H
#define KOKKOS_WRAPPER_PARAMETERS_H

// common parameters
#define VIEW_LAYOUTS $p_VIEW_LAYOUTS
#define VIEW_DIMENSIONS $p_VIEW_DIMS
#define VIEW_TYPES $p_VIEW_TYPES
#define EXEC_SPACE_FILTER $p_EXEC_SPACES
#define MEM_SPACE_FILTER $p_MEM_SPACES

// copy.cpp parameters
#define DEST_LAYOUTS $p_DEST_LAYOUTS
#define WITHOUT_EXEC_SPACE_ARG $p_WITHOUT_EXEC_SPACE_ARG

// mirrors.cpp parameters
#define DEST_MEM_SPACES $p_DEST_MEM_SPACES
#define WITH_NOTHING_ARG $p_WITH_NOTHING_ARG

// subviews.cpp parameters
#define SUBVIEW_DIMS $p_SUBVIEW_DIMS

#endif // KOKKOS_WRAPPER_PARAMETERS_H

END_OF_FILE
