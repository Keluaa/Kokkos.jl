
set(COMMON_HEADERS
        execution_spaces.h
        memory_spaces.h
        spaces.h
        layouts.h
        views.h
        copy.h
        mirrors.h
        subviews.h
        utils.h
        printing_utils.h)

add_custom_target(create_parameters
        COMMAND ${PROJECT_SOURCE_DIR}/build_parameters.sh
        BYPRODUCTS parameters.h
        DEPENDS ${PROJECT_SOURCE_DIR}/build_parameters.sh)

add_library(copy_lib SHARED copy.cpp ${COMMON_HEADERS})
target_include_directories(copy_lib PUBLIC ${PROJECT_BINARY_DIR})  # To include 'parameters.h'
target_link_libraries(copy_lib PUBLIC JlCxx::cxxwrap_julia)
target_link_libraries(copy_lib PRIVATE Kokkos::kokkos)
if(Kokkos_ENABLE_OPENMP)
        target_link_libraries(copy_lib PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(copy_lib PRIVATE
        COMPLETE_BUILD=0)

add_custom_target(compile_copy
        COMMAND cmake -E rename $<TARGET_FILE:copy_lib> ${PROJECT_BINARY_DIR}/libcopy_out.so
        BYPRODUCTS ${PROJECT_BINARY_DIR}/libcopy_out.so
        COMMAND_EXPAND_LISTS
        DEPENDS create_parameters copy_lib)
