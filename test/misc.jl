@testset "misc" begin

io = IOBuffer()
Kokkos.versioninfo(io; verbose=true)
str = String(take!(io))
@test occursin(string(Kokkos.KOKKOS_VERSION), str)
@test occursin("$(Kokkos.kokkos_name(TEST_BACKEND_DEVICE)) Runtime Configuration", str)
@test occursin("Serial Runtime Configuration", str)

io = IOBuffer()
Kokkos.configinfo(io) === nothing
str = String(take!(io))
@test occursin(string(Kokkos.KOKKOS_VERSION), str)

prev_view_dims = Kokkos.KOKKOS_VIEW_DIMS
@test_logs (:info, r"Restart your Julia session") Kokkos.set_view_dims([5, 6])
@test Kokkos.KOKKOS_VIEW_DIMS == prev_view_dims
@test_logs (:warn, r"Kokkos configuration changed!") Kokkos.warn_config_changed()

# Only present here to keep the expected configuration without extra logging
@test_logs (:info, r"Restart your Julia session") Kokkos.set_view_dims([1, 2])

end