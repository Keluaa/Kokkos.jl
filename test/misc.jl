@testset "Misc" begin

@testset "Formatting" begin
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
end


@testset "Config warnings" begin
    prev_backends = copy(Kokkos.KOKKOS_BACKENDS)
    @test_logs (:info, r"Restart your Julia session") Kokkos.set_backends([TEST_UNAVAILABLE_BACKEND])
    @test Kokkos.KOKKOS_BACKENDS == prev_backends
    @test_logs (:warn, r"Kokkos configuration changed!") Kokkos.warn_config_changed()

    # Only present here to keep the expected configuration without extra logging
    @test_logs (:info, r"Restart your Julia session") Kokkos.set_backends(prev_backends)
end


@testset "Version resolving" begin
    latest = Kokkos.Wrapper.get_latest_kokkos_release(nothing, nothing)
    @test VersionNumber(latest) ≥ v"4.1.0"
    @test Kokkos.Wrapper.get_latest_kokkos_release("latest") == latest

    latest_3 = Kokkos.Wrapper.get_latest_kokkos_release("3-latest")
    @test v"3.7.0" < VersionNumber(latest_3) < v"4.0.0"

    latest_37 = Kokkos.Wrapper.get_latest_kokkos_release("3.7-latest")
    @test v"3.7.0" < VersionNumber(latest_37) ≤ VersionNumber(latest_3)

    @test_throws r"No Kokkos release" Kokkos.Wrapper.get_latest_kokkos_release("99-latest")

    @test_throws r"Expected version" Kokkos.Wrapper.get_latest_kokkos_release("oops")
    @test_throws r"Expected version" Kokkos.Wrapper.get_latest_kokkos_release(".-latest")
end


Kokkos.DynamicCompilation.clean_libs()
@test isempty(readdir(Kokkos.Wrapper.get_kokkos_func_libs_dir()))

end
