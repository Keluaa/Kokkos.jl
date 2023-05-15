@testset "utils" begin
    
@test Kokkos.require(; version = ≥(v"4.0.0"), no_error = true) == (Kokkos.KOKKOS_VERSION ≥ v"4.0.0")
@test Kokkos.require(; version = <(v"4.0.0"), no_error = true) == (Kokkos.KOKKOS_VERSION < v"4.0.0")

@test Kokkos.require(; dims = [1], no_error = true)
@test Kokkos.require(; dims = [1, 2], no_error = true)
@test !Kokkos.require(; dims = [1, 2, 3], no_error = true)
@test !Kokkos.require(; dims = [3], no_error = true)

@test Kokkos.require(; exec_spaces = [TEST_BACKEND_DEVICE], no_error = true)
@test !Kokkos.require(; exec_spaces = [TEST_UNAVAILABLE_BACKEND], no_error = true)

@test Kokkos.require(; idx = (==(8) ∘ sizeof), no_error = true)

end