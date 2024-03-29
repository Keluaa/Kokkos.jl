
using Test
using Logging
using Preferences
using Kokkos

# All environment variables affecting tests are mentioned here.

const TEST_KOKKOS_VERSION = get(ENV, "TEST_KOKKOS_VERSION", "4.0.01")
const TEST_CMAKE_OPTIONS = filter!(!isempty, (split(get(ENV, "TEST_CMAKE_OPTIONS", ""), '|')))

const TEST_CUDA = parse(Bool, get(ENV, "TEST_KOKKOS_CUDA", "false"))
const TEST_HIP  = parse(Bool, get(ENV, "TEST_KOKKOS_HIP", "false"))
TEST_CUDA && TEST_HIP && error("Only a single GPU backend can be enabled at once")

const TEST_OPENMP = !(TEST_CUDA || TEST_HIP)
const TEST_DEVICE_IS_HOST = TEST_OPENMP

const TEST_MPI_ONLY = parse(Bool, get(ENV, "TEST_KOKKOS_MPI_ONLY", "false"))
const TEST_MPI = parse(Bool, get(ENV, "TEST_KOKKOS_MPI", "true")) || TEST_MPI_ONLY

if TEST_CUDA
    const TEST_BACKEND_HOST          = Kokkos.Serial
    const TEST_BACKEND_DEVICE        = Kokkos.Cuda
    const TEST_UNAVAILABLE_BACKEND   = Kokkos.HIP

    const TEST_MEM_SPACE_HOST        = Kokkos.HostSpace
    const TEST_MEM_SPACES_DEVICE     = (Kokkos.CudaSpace, Kokkos.CudaUVMSpace)
    const TEST_UNAVAILABLE_MEM_SPACE = Kokkos.HIPSpace

    const TEST_MEM_SHARED            = Kokkos.CudaUVMSpace
    const TEST_MEM_PINNED            = Kokkos.CudaHostPinnedSpace
elseif TEST_HIP
    const TEST_BACKEND_HOST          = Kokkos.Serial
    const TEST_BACKEND_DEVICE        = Kokkos.HIP
    const TEST_UNAVAILABLE_BACKEND   = Kokkos.Cuda

    const TEST_MEM_SPACE_HOST        = Kokkos.HostSpace
    const TEST_MEM_SPACES_DEVICE     = (Kokkos.HIPSpace, Kokkos.HIPManagedSpace)
    const TEST_UNAVAILABLE_MEM_SPACE = Kokkos.CudaSpace

    const TEST_MEM_SHARED            = Kokkos.HIPManagedSpace
    const TEST_MEM_PINNED            = Kokkos.HIPHostPinnedSpace
else
    const TEST_BACKEND_HOST          = Kokkos.Serial
    const TEST_BACKEND_DEVICE        = Kokkos.OpenMP
    const TEST_UNAVAILABLE_BACKEND   = Kokkos.Cuda

    const TEST_MEM_SPACE_HOST        = Kokkos.HostSpace
    const TEST_MEM_SPACES_DEVICE     = (Kokkos.HostSpace,)
    const TEST_UNAVAILABLE_MEM_SPACE = Kokkos.CudaSpace

    const TEST_MEM_SHARED            = Kokkos.HostSpace
    const TEST_MEM_PINNED            = Kokkos.HostSpace
end

const TEST_MAIN_MEM_SPACE_DEVICE = first(TEST_MEM_SPACES_DEVICE)

const TEST_DEVICE_ACCESSIBLE = !(TEST_CUDA || TEST_HIP)

const TEST_VIEW_DIMS = (1, 2)
const TEST_VIEW_TYPES = (Float64, Int64)
const TEST_VIEW_LAYOUTS = (Kokkos.LayoutLeft, Kokkos.LayoutRight, Kokkos.LayoutStride)


TEST_CUDA && using CUDA
TEST_HIP && using AMDGPU
TEST_MPI && using MPI


function print_test_config()
    println("Test configuration:")
    println(" - TEST_KOKKOS_VERSION:   $TEST_KOKKOS_VERSION")
    println(" - TEST_OPENMP:           $TEST_OPENMP")
    println(" - TEST_CUDA:             $TEST_CUDA")
    println(" - TEST_HIP:              $TEST_HIP")
    println(" - TEST_MPI:              $TEST_MPI (only MPI: $TEST_MPI_ONLY)")
    println(" - BACKEND_HOST:          $(nameof(TEST_BACKEND_HOST))")
    println(" - BACKEND_DEVICE:        $(nameof(TEST_BACKEND_DEVICE))")
    println(" - UNAVAILABLE_BACKEND:   $(nameof(TEST_UNAVAILABLE_BACKEND))")
    println(" - UNAVAILABLE_MEM_SPACE: $(nameof(TEST_UNAVAILABLE_MEM_SPACE))")
    println(" - MEM_SPACE_HOST:        $(nameof(TEST_MEM_SPACE_HOST))")
    println(" - MEM_SPACES_DEVICE:     $(join(nameof.(TEST_MEM_SPACES_DEVICE), ", "))")
    println(" - MAIN_MEM_SPACE_DEVICE: $(nameof(TEST_MAIN_MEM_SPACE_DEVICE))")
    println(" - MEM_SHARED:            $(nameof(TEST_MEM_SHARED))")
    println(" - MEM_PINNED:            $(nameof(TEST_MEM_PINNED))")
    println(" - VIEW_DIMS:             $(TEST_VIEW_DIMS)")
    println(" - VIEW_TYPES:            $(TEST_VIEW_TYPES)")
    println(" - VIEW_LAYOUTS:          $(TEST_VIEW_LAYOUTS)")

    @static if VERSION >= v"1.9-"
        TEST_CUDA && println(" - CUDA.jl:               ", pkgversion(CUDA))
        TEST_HIP  && println(" - AMDGPU.jl:             ", pkgversion(AMDGPU))
        TEST_MPI  && println(" - MPI.jl:                ", pkgversion(MPI))
    end
end


@testset "Kokkos.jl" begin
    print_test_config()

    include("pre_wrapper_load.jl")

    Kokkos.build_in_project()  # Use the same directory as Pkg.test uses, forcing a complete compilation

    Kokkos.set_omp_vars()
    if TEST_OPENMP
        @test_logs min_level=Logging.Warn @test_nowarn Kokkos.load_wrapper_lib(; loading_bar=false)
    else
        # GPU backends add some warnings which I can't get rid of
        @test_logs min_level=Logging.Warn Kokkos.load_wrapper_lib(; loading_bar=false)
    end

    if !TEST_MPI_ONLY
        @test_nowarn Kokkos.initialize()

        Kokkos.DynamicCompilation.clean_libs()  # Make sure no leftover lib influences the test results

        if !endswith(TEST_KOKKOS_VERSION, "latest") && Kokkos.KOKKOS_VERSION != VersionNumber(TEST_KOKKOS_VERSION)
            error("Expected Kokkos v$TEST_KOKKOS_VERSION, got: $(Kokkos.KOKKOS_VERSION)")
        end

        if [TEST_BACKEND_HOST, TEST_BACKEND_DEVICE] ⊈ Kokkos.ENABLED_EXEC_SPACES
            error("Invalid execution spaces: $([TEST_BACKEND_HOST, TEST_BACKEND_DEVICE]) ⊈ $(Kokkos.ENABLED_EXEC_SPACES)")
        end

        include("spaces.jl")

        if TEST_DEVICE_ACCESSIBLE
            include("views.jl")
        else
            include("views_gpu.jl")
        end

        if TEST_CUDA
            include("backends/cuda.jl")
        elseif TEST_HIP
            include("backends/hip.jl")
        end

        include("projects.jl")
        include("simple_lib_tests.jl")
    end

    TEST_MPI && include("mpi.jl")

    @info "Tests needed $(length(Kokkos.DynamicCompilation.LOADED_FUNCTION_LIBS)) function libraries"

    if !TEST_MPI_ONLY
        include("misc.jl")

        @test_nowarn Kokkos.finalize()  # Let the finalize hook call the finalizers of all views
        GC.gc(true)  # Any view which calls its finalizer here should throw
    end
end
