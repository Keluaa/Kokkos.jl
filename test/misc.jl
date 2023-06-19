@testset "Misc" begin

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

prev_backends = copy(Kokkos.KOKKOS_BACKENDS)
@test_logs (:info, r"Restart your Julia session") Kokkos.set_backends([TEST_UNAVAILABLE_BACKEND])
@test Kokkos.KOKKOS_BACKENDS == prev_backends
@test_logs (:warn, r"Kokkos configuration changed!") Kokkos.warn_config_changed()

# Only present here to keep the expected configuration without extra logging
@test_logs (:info, r"Restart your Julia session") Kokkos.set_backends(prev_backends)

# TODO: test Kokkos.DynamicCompilation.clean_libs()

end
