@testset "utils" begin
    
@test Kokkos.require(; version = >=(v"4.0.0"), no_error = true)
@test !Kokkos.require(; version = <(v"4.0.0"), no_error = true)

@test Kokkos.require(; dims = [1], no_error = true)
@test Kokkos.require(; dims = [1, 2], no_error = true)
@test !Kokkos.require(; dims = [1, 2, 3], no_error = true)
@test !Kokkos.require(; dims = [3], no_error = true)

@test Kokkos.require(; exec_spaces = [Kokkos.OpenMP], no_error = true)
@test !Kokkos.require(; exec_spaces = [Kokkos.Cuda], no_error = true)

@test Kokkos.require(; idx = (==(8) ∘ sizeof), no_error = true)

end