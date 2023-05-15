@testset "spaces" begin

@test Kokkos.COMPILED_EXEC_SPACES == tuple(unique((TEST_BACKEND_HOST, TEST_BACKEND_DEVICE))...)
@test Kokkos.DEFAULT_DEVICE_SPACE === TEST_BACKEND_DEVICE
@test Kokkos.DEFAULT_HOST_SPACE === TEST_BACKEND_DEVICE

@test Kokkos.COMPILED_MEM_SPACES == tuple(unique((TEST_MEM_SPACE_HOST, TEST_MEM_SPACES_DEVICE..., TEST_MEM_SHARED, TEST_MEM_PINNED))...)
@test Kokkos.DEFAULT_DEVICE_MEM_SPACE === TEST_MAIN_MEM_SPACE_DEVICE
@test Kokkos.DEFAULT_HOST_MEM_SPACE === TEST_MEM_SPACE_HOST

skip_shared_mem = Kokkos.KOKKOS_VERSION < v"4.0.0"
@test Kokkos.SHARED_MEMORY_SPACE === TEST_MEM_SHARED skip=skip_shared_mem
@test Kokkos.SHARED_HOST_PINNED_MEMORY_SPACE === TEST_MEM_PINNED skip=skip_shared_mem


@testset "enabled" begin
    for exec_space in Kokkos.Spaces.ALL_BACKENDS
        @test Kokkos.enabled(exec_space) == (exec_space in Kokkos.COMPILED_EXEC_SPACES)
    end

    for mem_space in Kokkos.Spaces.ALL_MEM_SPACES
        @test Kokkos.enabled(mem_space) == (mem_space in Kokkos.COMPILED_MEM_SPACES)
    end
end


@test Kokkos.execution_space(TEST_MEM_SPACE_HOST) === (TEST_OPENMP ? Kokkos.OpenMP : TEST_BACKEND_HOST)
@test Kokkos.memory_space(TEST_BACKEND_HOST) === TEST_MEM_SPACE_HOST
@test Kokkos.memory_space(TEST_BACKEND_DEVICE) === TEST_MAIN_MEM_SPACE_DEVICE


@testset "accessible" begin
    @test Kokkos.accessible(TEST_MEM_SPACE_HOST)
    @test Kokkos.accessible(TEST_BACKEND_HOST, TEST_MEM_SPACE_HOST)
    @test Kokkos.accessible(TEST_MEM_SPACE_HOST, TEST_MEM_SPACE_HOST)
    for mem_space in TEST_MEM_SPACES_DEVICE
        @test Kokkos.accessible(TEST_BACKEND_DEVICE, mem_space)
    end
end


@test Kokkos.array_layout(Kokkos.Serial) === Kokkos.LayoutRight

@test Kokkos.kokkos_name(Kokkos.Serial) == "Serial"
@test Kokkos.kokkos_name(Kokkos.HostSpace) == "Host"

@test Kokkos.main_space_type(Kokkos.Serial) === Kokkos.Serial
@test Kokkos.main_space_type(Kokkos.KokkosWrapper.Impl.SerialImpl) === Kokkos.Serial
@test Kokkos.main_space_type(Kokkos.KokkosWrapper.Impl.SerialImplAllocated) === Kokkos.Serial
@test Kokkos.main_space_type(Kokkos.KokkosWrapper.Impl.SerialImplDereferenced) === Kokkos.Serial

@test_throws @error_match("must be a subtype") Kokkos.main_space_type(Kokkos.Space)
@test_throws @error_match("must be a subtype") Kokkos.main_space_type(Kokkos.MemorySpace)
@test_throws @error_match("must be a subtype") Kokkos.main_space_type(Kokkos.ExecutionSpace)

@test Kokkos.impl_space_type(Kokkos.Serial) === Kokkos.KokkosWrapper.Impl.SerialImpl
@test Kokkos.impl_space_type(Kokkos.HostSpace) === Kokkos.KokkosWrapper.Impl.HostSpaceImpl

@test_throws @error_match("is not compiled") Kokkos.impl_space_type(TEST_UNAVAILABLE_BACKEND)

serial = Kokkos.Serial()
@test Kokkos.main_space_type(serial) === Kokkos.Serial
@test Kokkos.memory_space(serial) === Kokkos.HostSpace
@test Kokkos.enabled(serial)
@test Kokkos.array_layout(serial) === Kokkos.LayoutRight

host_space = Kokkos.HostSpace()
@test Kokkos.main_space_type(host_space) === Kokkos.HostSpace
@test Kokkos.execution_space(host_space) === (TEST_OPENMP ? Kokkos.OpenMP : TEST_BACKEND_HOST)
@test Kokkos.accessible(host_space)
@test Kokkos.enabled(host_space)

@test Kokkos.fence() === nothing
@test Kokkos.fence("test_fence") === nothing
@test Kokkos.fence(serial) === nothing
@test Kokkos.fence(serial, "test_fence_Serial") === nothing

@test Kokkos.concurrency(serial) == 1
@test Kokkos.concurrency(Kokkos.OpenMP()) == Threads.nthreads() skip=!TEST_OPENMP

alloc_ptr = Kokkos.allocate(host_space, 10)
@test alloc_ptr !== C_NULL
Kokkos.deallocate(host_space, alloc_ptr, 10)

end