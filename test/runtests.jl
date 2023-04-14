
cd(joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Kokkos
using Test
using Logging


macro error_match(exception)
    # To make @test_throws work in most cases in version 1.7 and above
    if exception isa Symbol
        return exception
    elseif VERSION ≥ v"1.8"
        return exception
    else
        return ErrorException
    end
end


@testset "Kokkos.jl" begin
    include("pre_wrapper_load.jl")

    Kokkos.set_omp_vars()
    @test_logs min_level=Logging.Warn @test_nowarn Kokkos.load_wrapper_lib()
    @test_nowarn Kokkos.initialize()
    Kokkos.require(; dims=[1, 2], types=[Float64, Int64], exec_spaces=[Kokkos.Serial, Kokkos.OpenMP])

    include("views.jl")
    include("spaces.jl")
    include("utils.jl")
    include("simple_lib_tests.jl")
    include("misc.jl")
end
