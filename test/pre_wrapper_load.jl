@testset "Pre-load" begin

Kokkos.is_initialized() && error("Kokkos should not be loaded for those tests")

@test !Kokkos.is_initialized()
@test !Kokkos.is_finalized()

@test_throws @error_match("is not loaded") Kokkos.ensure_kokkos_wrapper_loaded()
@test_throws @error_match("is not loaded") Kokkos.new_initialization_settings()
@test_throws @error_match("is not loaded") Kokkos.versioninfo()
@test_throws @error_match("is not loaded") Kokkos.require()
@test_throws @error_match("is not loaded") Kokkos.View{Float64}()

default_path = Kokkos.LOCAL_KOKKOS_DIR
@test Kokkos.set_kokkos_version("3.7.01")                       == Kokkos.LOCAL_KOKKOS_VERSION_STR == "3.7.01"
@test Kokkos.set_kokkos_path(default_path)                      == Kokkos.KOKKOS_PATH              == default_path
@test Kokkos.set_cmake_options(["-DCMAKE_BUILD_TYPE=Debug"])    == Kokkos.KOKKOS_CMAKE_OPTIONS     == ["-DCMAKE_BUILD_TYPE=Debug"]
@test Kokkos.set_backends([Kokkos.Serial, Kokkos.Cuda])         == Kokkos.KOKKOS_BACKENDS          == ["Serial", "Cuda"]
@test Kokkos.set_view_dims([3, 4, 5])                           == Kokkos.KOKKOS_VIEW_DIMS         == [3, 4, 5]
@test Kokkos.set_view_types([Bool, Float32, Int16])             == Kokkos.KOKKOS_VIEW_TYPES        == ["Bool", "Float32", "Int16"]
@test Kokkos.set_build_type("Debug")                            == Kokkos.KOKKOS_BUILD_TYPE        == "Debug"
@test Kokkos.set_build_dir(joinpath(@__DIR__, ".kokkos-build")) == Kokkos.KOKKOS_BUILD_DIR         == joinpath(@__DIR__, ".kokkos-build")
@test Kokkos.set_kokkos_options(Dict("Kokkos_ENABLE_CUDA_CONSTEXPR" => "ON")) == Kokkos.KOKKOS_LIB_OPTIONS == ["Kokkos_ENABLE_CUDA_CONSTEXPR=ON"]
@test Kokkos.set_kokkos_options(["Kokkos_ENABLE_CUDA_CONSTEXPR=ON"])          == Kokkos.KOKKOS_LIB_OPTIONS == ["Kokkos_ENABLE_CUDA_CONSTEXPR=ON"]

@test Kokkos.set_kokkos_version(missing) == Kokkos.LOCAL_KOKKOS_VERSION_STR == Kokkos.__DEFAULT_KOKKOS_VERSION_STR
@test Kokkos.set_kokkos_path(missing)    == Kokkos.KOKKOS_PATH              == Kokkos.LOCAL_KOKKOS_DIR
@test Kokkos.set_cmake_options(missing)  == Kokkos.KOKKOS_CMAKE_OPTIONS     == Kokkos.__DEFAULT_KOKKOS_CMAKE_OPTIONS
@test Kokkos.set_kokkos_options(missing) == Kokkos.KOKKOS_LIB_OPTIONS       == Kokkos.__DEFAULT_KOKKOS_LIB_OPTIONS
@test Kokkos.set_build_type(missing)     == Kokkos.KOKKOS_BUILD_TYPE        == Kokkos.__DEFAULT_KOKKOS_BUILD_TYPE
@test Kokkos.set_build_dir(missing)      == Kokkos.KOKKOS_BUILD_DIR         == Kokkos.__DEFAULT_KOKKOS_BUILD_DIR

if VERSION ≥ v"1.8"
    # In 1.8 and above, `Pkg.test` has a different behaviour from `include("./runtests.jl")`, which
    # changes how default values of preferences are affected by the package's "LocalPreferences.toml"
    @test Kokkos.set_backends(missing)   == Kokkos.KOKKOS_BACKENDS   == ["Serial", "OpenMP"]
    @test Kokkos.set_view_dims(missing)  == Kokkos.KOKKOS_VIEW_DIMS  == [1, 2]
    @test Kokkos.set_view_types(missing) == Kokkos.KOKKOS_VIEW_TYPES == ["Float64", "Int64"]
else
    @test Kokkos.set_backends(missing)   == Kokkos.KOKKOS_BACKENDS   == Kokkos.__DEFAULT_KOKKOS_BACKENDS
    @test Kokkos.set_view_dims(missing)  == Kokkos.KOKKOS_VIEW_DIMS  == Kokkos.__DEFAULT_KOKKOS_VIEW_DIMS
    @test Kokkos.set_view_types(missing) == Kokkos.KOKKOS_VIEW_TYPES == Kokkos.__DEFAULT_KOKKOS_VIEW_TYPES
end

# Set the configuration back to what other tests expect (on top of the default values)
Kokkos.set_backends([Kokkos.Serial, Kokkos.OpenMP])
Kokkos.set_view_dims([1, 2])
Kokkos.set_view_types([Float64, Int64])

end