
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

    Kokkos.build_in_project()  # Use the same directory as Pkg.test uses, forcing a complete compilation

    Kokkos.set_omp_vars()
    @test_logs min_level=Logging.Warn @test_nowarn Kokkos.load_wrapper_lib(; loading_bar=false)
    @test_nowarn Kokkos.initialize()
    Kokkos.require(;
        dims=[1, 2],
        types=[Float64, Int64],
        layouts=[Kokkos.LayoutLeft, Kokkos.LayoutRight, Kokkos.LayoutStride],
        exec_spaces=[Kokkos.Serial, Kokkos.OpenMP]
    )

    include("views.jl")
    include("spaces.jl")
    include("utils.jl")
    include("projects.jl")
    include("simple_lib_tests.jl")
    include("misc.jl")
    include("mpi.jl")

    GC.gc(true)  # Call the finalizers of all created views
    Kokkos.finalize()
end
