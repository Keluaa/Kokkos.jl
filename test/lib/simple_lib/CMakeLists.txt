cmake_minimum_required(VERSION 3.20)
project(SimpleKokkosLib)

set(CMAKE_CXX_STANDARD 17)

find_package(Kokkos REQUIRED)

add_library(SimpleKokkosLib1D SHARED
    simple_lib_1D.cpp
)
target_link_libraries(SimpleKokkosLib1D PRIVATE Kokkos::kokkos)

add_library(SimpleKokkosLib2D SHARED
    simple_lib_2D.cpp
)
target_link_libraries(SimpleKokkosLib2D PRIVATE Kokkos::kokkos)
