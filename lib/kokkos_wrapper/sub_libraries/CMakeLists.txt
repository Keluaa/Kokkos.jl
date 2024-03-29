
set(COMMON_HEADERS
        views.h
        ../parameters.h
        ../spaces.h ../execution_spaces.h ../memory_spaces.h
        ../layouts.h
        ../utils.h ../printing_utils.h ../kokkos_utils.h)


add_custom_target(create_parameters
        COMMAND ${PROJECT_SOURCE_DIR}/build_parameters.sh
        BYPRODUCTS ${PROJECT_BINARY_DIR}/build_parameters.h
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        DEPENDS ${PROJECT_SOURCE_DIR}/build_parameters.sh)


function(add_dynamic_compilation_library name source_file)
    add_library(${name} SHARED EXCLUDE_FROM_ALL ${source_file} ${COMMON_HEADERS})
    target_include_directories(${name} PUBLIC
            ${PROJECT_SOURCE_DIR}  # To include the wrapper's headers
            ${PROJECT_BINARY_DIR}  # To include 'build_parameters.h'
    )
    target_link_libraries(${name} PUBLIC JlCxx::cxxwrap_julia Kokkos::kokkos)
    if(Kokkos_ENABLE_OPENMP)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(${name} PRIVATE
            $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
    )
endfunction()


function(add_compilation_target name lib_name lib_output)
    # Clean all files of the target before compiling (but not the Kokkos files, which should be compiled once alongside
    # with the main wrapper library). Not doing so systematically can create errors tricky to debug, such as
    # methods failing to be specified, double registration of some types, etc...
    # TODO: incompatible with Ninja, which uses another way for cleaning targets (use CMAKE_GENERATOR to detect Ninja)
    add_custom_target(clean_${lib_name}
            COMMAND ${CMAKE_COMMAND} -P ${PROJECT_BINARY_DIR}/sub_libraries/CMakeFiles/${lib_name}.dir/cmake_clean.cmake
            COMMENT "Cleaning files of ${lib_name}")

    # Target to create the 'build_parameters.h', clean any previous build dir, build the lib and move it to the main
    # binary directory. Because of this last step, the target file is never up-to-date, this target can therefore be ran
    # as many times as needed, with different environment variables for 'build_parameters.h' each time.
    add_custom_target(${name}
            COMMAND cmake -E rename $<TARGET_FILE:${lib_name}> ${PROJECT_BINARY_DIR}/${lib_output}${CMAKE_SHARED_LIBRARY_SUFFIX}
            BYPRODUCTS ${PROJECT_BINARY_DIR}/${lib_output}${CMAKE_SHARED_LIBRARY_SUFFIX}
            COMMAND_EXPAND_LISTS
            DEPENDS create_parameters clean_${lib_name} ${lib_name})
endfunction()


add_dynamic_compilation_library(views_lib views.cpp)
add_compilation_target(views views_lib libviews_out)

add_dynamic_compilation_library(copy_lib copy.cpp)
add_compilation_target(copy copy_lib libcopy_out)

add_dynamic_compilation_library(subviews_lib subviews.cpp)
add_compilation_target(subviews subviews_lib libsubviews_out)

add_dynamic_compilation_library(mirrors_lib mirrors.cpp)
add_compilation_target(mirrors mirrors_lib libmirrors_out)
