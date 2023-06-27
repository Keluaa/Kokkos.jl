@testset "Misc" begin

io = IOBuffer()
Kokkos.versioninfo(io; verbose=true)
str = String(take!(io))
@test occursin(string(Kokkos.KOKKOS_VERSION), str)
if !(TEST_BACKEND_DEVICE in (Kokkos.HIP, Kokkos.SYCL))
    # There is only 2 Kokkos backends which are not formatted this way...
    @test occursin("$(Kokkos.kokkos_name(TEST_BACKEND_DEVICE)) Runtime Configuration", str)
end
@test occursin("Serial Runtime Configuration", str)

io = IOBuffer()
Kokkos.configinfo(io) === nothing
str = String(take!(io))
@test occursin(string(Kokkos.LOCAL_KOKKOS_VERSION_STR), str)

prev_backends = copy(Kokkos.KOKKOS_BACKENDS)
@test_logs (:info, r"Restart your Julia session") Kokkos.set_backends([TEST_UNAVAILABLE_BACKEND])
@test Kokkos.KOKKOS_BACKENDS == prev_backends
@test_logs (:warn, r"Kokkos configuration changed!") Kokkos.warn_config_changed()

# Only present here to keep the expected configuration without extra logging
@test_logs (:info, r"Restart your Julia session") Kokkos.set_backends(prev_backends)

Kokkos.DynamicCompilation.clean_libs()
@test isempty(readdir(Kokkos.Wrapper.get_kokkos_func_libs_dir()))

end
