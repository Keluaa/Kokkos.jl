cmake_minimum_required(VERSION 3.22)
project(KokkosWrapper)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
    # Required to be OFF when using Clang CUDA to avoid errors like '__float128 is not supported on this target'
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(NOT DEFINED CMAKE_INSTALL_MODE)
    # Symlinks are faster to create than copies. This should reduce initialization time.
    set(CMAKE_INSTALL_MODE ABS_SYMLINK_OR_COPY)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Needed to include the Kokkos libs in the shared library
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(JlCxx)


if(Kokkos_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()


if(Kokkos_ENABLE_CUDA)
    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON)
    if (NOT DEFINED Kokkos_ENABLE_CUDA_LAMBDA)
        set(Kokkos_ENABLE_CUDA_LAMBDA ON)
    endif()
endif()


if(CMAKE_BUILD_TYPE EQUAL "Debug")
    if (NOT DEFINED Kokkos_ENABLE_DEBUG)
        set(Kokkos_ENABLE_DEBUG ON)
    endif()
    if (NOT DEFINED Kokkos_ENABLE_DEBUG_BOUNDS_CHECK)
        set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON)
    endif()
    if (NOT DEFINED Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK)
        set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON)
    endif()
endif()


add_subdirectory(${Kokkos_ROOT} lib/kokkos)


set(WRAPPER_SOURCES
        kokkos_wrapper.cpp kokkos_wrapper.h
        execution_spaces.h
        memory_spaces.h
        spaces.cpp spaces.h
        space_specific_methods.cpp
        layouts.cpp layouts.h
        views.h
        copy.h
        mirrors.h
        subviews.h
        utils.h printing_utils.h kokkos_utils.h)


add_library(KokkosWrapper SHARED ${WRAPPER_SOURCES})
target_link_libraries(KokkosWrapper PUBLIC JlCxx::cxxwrap_julia Kokkos::kokkos)
target_compile_definitions(KokkosWrapper PRIVATE WRAPPER_BUILD)
target_compile_options(KokkosWrapper PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
)

if(Kokkos_ENABLE_OPENMP)
    target_link_libraries(KokkosWrapper PRIVATE OpenMP::OpenMP_CXX)
endif()

include(all_targets_generator.cmake)
