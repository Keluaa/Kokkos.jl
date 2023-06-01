
# Remove quotes, as they should not appear in macros
p_VIEW_LAYOUTS=$(echo $VIEW_LAYOUTS | tr -d '"')
p_VIEW_DIMS=$(echo $VIEW_DIMS | tr -d '"')
p_VIEW_TYPES=$(echo $VIEW_TYPES | tr -d '"')
p_EXEC_SPACES=$EXEC_SPACES
p_MEM_SPACES=$MEM_SPACES
p_WITHOUT_EXEC_SPACE_ARG=$(echo $WITHOUT_EXEC_SPACE_ARG | tr -d '"')


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
#define WITHOUT_EXEC_SPACE_ARG $p_WITHOUT_EXEC_SPACE_ARG

#endif // KOKKOS_WRAPPER_PARAMETERS_H
END_OF_FILE
