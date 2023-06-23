
set(COMMON_HEADERS
        execution_spaces.h
        memory_spaces.h
        spaces.h
        layouts.h
        views.h
        copy.h
        mirrors.h
        subviews.h
        utils.h printing_utils.h Kokkos_utils.h)


add_custom_target(create_parameters
        COMMAND ${PROJECT_SOURCE_DIR}/build_parameters.sh
        BYPRODUCTS parameters.h
        DEPENDS ${PROJECT_SOURCE_DIR}/build_parameters.sh)


function(add_dynamic_compilation_library name source_file)
    add_library(${name} SHARED EXCLUDE_FROM_ALL ${source_file} ${COMMON_HEADERS})
    target_include_directories(${name} PUBLIC ${PROJECT_BINARY_DIR})  # To include 'parameters.h'
    target_link_libraries(${name} PUBLIC JlCxx::cxxwrap_julia Kokkos::kokkos)
    if(Kokkos_ENABLE_OPENMP)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(KokkosWrapper PRIVATE
            $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
    )
    target_compile_definitions(${name} PRIVATE
            COMPLETE_BUILD=0
    )
endfunction()


function(add_compilation_target name lib_name lib_output)
    add_custom_target(${name}
            COMMAND cmake -E rename $<TARGET_FILE:${lib_name}> ${PROJECT_BINARY_DIR}/${lib_output}${CMAKE_SHARED_LIBRARY_SUFFIX}
            BYPRODUCTS ${PROJECT_BINARY_DIR}/${lib_output}${CMAKE_SHARED_LIBRARY_SUFFIX}
            COMMAND_EXPAND_LISTS
            DEPENDS create_parameters ${lib_name})
endfunction()


add_dynamic_compilation_library(views_lib views.cpp)
add_compilation_target(views views_lib libviews_out)

add_dynamic_compilation_library(copy_lib copy.cpp)
add_compilation_target(copy copy_lib libcopy_out)

add_dynamic_compilation_library(subviews_lib subviews.cpp)
add_compilation_target(subviews subviews_lib libsubviews_out)

add_dynamic_compilation_library(mirrors_lib mirrors.cpp)
add_compilation_target(mirrors mirrors_lib libmirrors_out)
