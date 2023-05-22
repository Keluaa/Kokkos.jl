
using Kokkos
using Test
using Logging


const TEST_CUDA = parse(Bool, get(ENV, "TEST_KOKKOS_CUDA", "false"))
const TEST_OPENMP = !TEST_CUDA
const TEST_DEVICE_IS_HOST = TEST_OPENMP

const TEST_BACKEND_HOST        = Kokkos.Serial
const TEST_BACKEND_DEVICE      = TEST_CUDA ? Kokkos.Cuda : Kokkos.OpenMP
const TEST_UNAVAILABLE_BACKEND = TEST_CUDA ? Kokkos.HIP  : Kokkos.Cuda

const TEST_MEM_SPACE_HOST = Kokkos.HostSpace
const TEST_MEM_SPACES_DEVICE = TEST_CUDA ? (Kokkos.CudaSpace, Kokkos.CudaUVMSpace) : (Kokkos.HostSpace,)
const TEST_MAIN_MEM_SPACE_DEVICE = first(TEST_MEM_SPACES_DEVICE)

const TEST_MEM_SHARED = TEST_CUDA ? Kokkos.CudaUVMSpace        : Kokkos.HostSpace
const TEST_MEM_PINNED = TEST_CUDA ? Kokkos.CudaHostPinnedSpace : Kokkos.HostSpace

const TEST_DEVICE_ACCESSIBLE = !TEST_CUDA
const TEST_IDX_SIZE = TEST_DEVICE_IS_HOST ? 8 : 4


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
        exec_spaces=[TEST_BACKEND_HOST, TEST_BACKEND_DEVICE]
    )

    include("spaces.jl")

    if TEST_DEVICE_ACCESSIBLE
        include("views.jl")
    else
        include("views_gpu.jl")
    end

    if TEST_CUDA
        include("backends/cuda.jl")
    end

    include("utils.jl")
    include("projects.jl")
    include("simple_lib_tests.jl")
    include("misc.jl")
    include("mpi.jl")

    GC.gc(true)  # Call the finalizers of all created views
    @test_nowarn Kokkos.finalize()
end
