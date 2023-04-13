@testset "spaces" begin

@test Kokkos.COMPILED_EXEC_SPACES == (Kokkos.Serial, Kokkos.OpenMP)
@test Kokkos.DEFAULT_DEVICE_SPACE === Kokkos.OpenMP
@test Kokkos.DEFAULT_HOST_SPACE === Kokkos.OpenMP

@test Kokkos.COMPILED_MEM_SPACES == (Kokkos.HostSpace,)
@test Kokkos.DEFAULT_DEVICE_MEM_SPACE === Kokkos.HostSpace
@test Kokkos.DEFAULT_HOST_MEM_SPACE === Kokkos.HostSpace

skip_shared_mem = Kokkos.KOKKOS_VERSION < v"4.0.0"
# Since host == device, both of these are defined to HostSpace
@test Kokkos.SHARED_MEMORY_SPACE === Kokkos.HostSpace skip=skip_shared_mem
@test Kokkos.SHARED_HOST_PINNED_MEMORY_SPACE === Kokkos.HostSpace skip=skip_shared_mem

@test Kokkos.enabled(Kokkos.Serial)
@test Kokkos.enabled(Kokkos.OpenMP)
@test !Kokkos.enabled(Kokkos.OpenACC)
@test !Kokkos.enabled(Kokkos.OpenMPTarget)
@test !Kokkos.enabled(Kokkos.Spaces.Threads)
@test !Kokkos.enabled(Kokkos.Cuda)
@test !Kokkos.enabled(Kokkos.HIP)
@test !Kokkos.enabled(Kokkos.HPX)
@test !Kokkos.enabled(Kokkos.SYCL)

@test Kokkos.enabled(Kokkos.HostSpace)
@test !Kokkos.enabled(Kokkos.CudaSpace)
@test !Kokkos.enabled(Kokkos.CudaHostPinnedSpace)
@test !Kokkos.enabled(Kokkos.CudaUVMSpace)
@test !Kokkos.enabled(Kokkos.HIPSpace)
@test !Kokkos.enabled(Kokkos.HIPHostPinnedSpace)
@test !Kokkos.enabled(Kokkos.HIPManagedSpace)

@test Kokkos.execution_space(Kokkos.HostSpace) === Kokkos.OpenMP
@test Kokkos.memory_space(Kokkos.Serial) === Kokkos.HostSpace
@test Kokkos.memory_space(Kokkos.OpenMP) === Kokkos.HostSpace

@test Kokkos.accessible(Kokkos.HostSpace)

@test Kokkos.kokkos_name(Kokkos.Serial) == "Serial"
@test Kokkos.kokkos_name(Kokkos.OpenMP) == "OpenMP"
@test Kokkos.kokkos_name(Kokkos.HostSpace) == "Host"

@test Kokkos.main_space_type(Kokkos.Serial) === Kokkos.Serial
@test Kokkos.main_space_type(Kokkos.Spaces.SerialImpl) === Kokkos.Serial
@test Kokkos.main_space_type(Kokkos.Spaces.SerialImplAllocated) === Kokkos.Serial
@test Kokkos.main_space_type(Kokkos.Spaces.SerialImplDereferenced) === Kokkos.Serial

if VERSION >= v"1.8"
    @test_throws "must be a subtype" Kokkos.main_space_type(Kokkos.Space)
    @test_throws "must be a subtype" Kokkos.main_space_type(Kokkos.MemorySpace)
    @test_throws "must be a subtype" Kokkos.main_space_type(Kokkos.ExecutionSpace)
end

serial = Kokkos.Spaces.SerialImpl()
@test Kokkos.main_space_type(serial) === Kokkos.Serial
@test Kokkos.memory_space(serial) === Kokkos.HostSpace
@test Kokkos.enabled(serial)

host_space = Kokkos.Spaces.HostSpaceImpl()
@test Kokkos.main_space_type(host_space) === Kokkos.HostSpace
@test Kokkos.execution_space(host_space) === Kokkos.OpenMP
@test Kokkos.accessible(host_space)
@test Kokkos.enabled(host_space)

end