cmake_minimum_required(VERSION 3.20)
project(KokkosWrapper)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
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
    set(VIEW_TYPES "double,float,int64_t")
endif()


add_library(KokkosWrapper SHARED
        kokkos_wrapper.cpp kokkos_wrapper.h
        execution_spaces.h
        memory_spaces.h
        spaces.cpp spaces.h
        views.cpp views.h
        copy.cpp copy.h)
target_link_libraries(KokkosWrapper PUBLIC JlCxx::cxxwrap_julia)
target_link_libraries(KokkosWrapper PRIVATE Kokkos::kokkos)
target_compile_definitions(KokkosWrapper PRIVATE
        VIEW_DIMENSIONS=${VIEW_DIMENSIONS}
        VIEW_TYPES=${VIEW_TYPES}
)
target_compile_options(KokkosWrapper PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
)

if(Kokkos_ENABLE_OPENMP)
    target_link_libraries(KokkosWrapper PRIVATE OpenMP::OpenMP_CXX)
endif()
