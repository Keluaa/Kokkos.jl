@testset "Projects" begin

lib_src = joinpath(@__DIR__, "lib", "simple_lib")
lib_build = joinpath(Kokkos.KOKKOS_BUILD_DIR, "simple_lib")
lib_name = "libSimpleKokkosLib1D"

project_1D = CMakeKokkosProject(lib_src, lib_name;
    target="SimpleKokkosLib1D", build_dir=lib_build)

@test Kokkos.source_dir(project_1D) == lib_src
@test Kokkos.build_dir(project_1D) == lib_build
@test Kokkos.lib_path(project_1D) == joinpath(lib_build, lib_name)

@test_nowarn Base.show(IOBuffer(), project_1D)

Kokkos.configuration_changed!(project_1D, false)
@test Kokkos.option!(project_1D, "ENABLE_CUDA_CONSTEXPR", true)
@test Kokkos.options(project_1D)["Kokkos_ENABLE_CUDA_CONSTEXPR"] == "ON"
@test Kokkos.option!(project_1D, "ENABLE_CUDA_CONSTEXPR", "OFF")
@test Kokkos.options(project_1D)["Kokkos_ENABLE_CUDA_CONSTEXPR"] == "OFF"
@test Kokkos.configuration_changed(project_1D)

end