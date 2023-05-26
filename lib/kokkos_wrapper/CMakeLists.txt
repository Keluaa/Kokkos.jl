cmake_minimum_required(VERSION 3.20)
project(KokkosWrapper)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Needed to include the Kokkos libs in the shared library
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(JlCxx)

if(Kokkos_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

add_subdirectory(${Kokkos_ROOT} lib/kokkos)

if(NOT DEFINED VIEW_DIMENSIONS)
    set(VIEW_DIMENSIONS "1,2")
endif()

if(NOT DEFINED VIEW_TYPES)
    set(VIEW_TYPES "double,int64_t")
endif()

if (NOT DEFINED VIEW_LAYOUTS)
    set(VIEW_LAYOUTS "deviceDefault,hostDefault")
endif()


add_library(KokkosWrapper SHARED
        kokkos_wrapper.cpp kokkos_wrapper.h
        execution_spaces.h
        memory_spaces.h
        spaces.cpp spaces.h
        space_specific_methods.cpp
        layouts.cpp layouts.h
        views.cpp views.h
        copy.cpp copy.h
        mirrors.cpp mirrors.h
        subviews.cpp subviews.h
        utils.h printing_utils.h)
target_link_libraries(KokkosWrapper PUBLIC JlCxx::cxxwrap_julia)
target_link_libraries(KokkosWrapper PRIVATE Kokkos::kokkos)
target_compile_definitions(KokkosWrapper PRIVATE
        VIEW_DIMENSIONS=${VIEW_DIMENSIONS}
        VIEW_TYPES=${VIEW_TYPES}
        VIEW_LAYOUTS=${VIEW_LAYOUTS}
        COMPLETE_BUILD=1
)
target_compile_options(KokkosWrapper PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
)

if(Kokkos_ENABLE_OPENMP)
    target_link_libraries(KokkosWrapper PRIVATE OpenMP::OpenMP_CXX)
endif()

include(all_targets_generator.cmake)
