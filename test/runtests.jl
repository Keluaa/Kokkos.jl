using Kokkos
using Test

Kokkos.require(; dims=[1, 2], types=[Float64, Int64], exec_spaces=[Kokkos.Serial, Kokkos.OpenMP])
Kokkos.set_omp_vars()
!Kokkos.is_initialized() && Kokkos.initialize()

@testset "Kokkos.jl" begin
    include("views.jl")
    include("spaces.jl")
    include("utils.jl")
    include("simple_lib_tests.jl")
end
