
# Remove quotes, as they cannot appear in macros
p_VIEW_LAYOUTS=$(echo $VIEW_LAYOUTS | tr -d '"')
p_VIEW_DIMS=$(echo $VIEW_DIMS | tr -d '"')
p_VIEW_TYPES=$(echo $VIEW_TYPES | tr -d '"')

cat > ./parameters.h <<- END_OF_FILE

#ifndef PARAMETERS_H
#define PARAMETERS_H

#define VIEW_LAYOUTS $p_VIEW_LAYOUTS
#define VIEW_DIMENSIONS $p_VIEW_DIMS
#define VIEW_TYPES $p_VIEW_TYPES

#endif // PARAMETERS_H
END_OF_FILE
