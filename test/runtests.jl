
push!(LOAD_PATH, joinpath(@__DIR__, ".."))  # For LocalPreferences.toml

using Test
using Logging
using Preferences
using Kokkos

# All environment variables affecting tests are mentioned here.

const TEST_CUDA = parse(Bool, get(ENV, "TEST_KOKKOS_CUDA", "false"))
const TEST_OPENMP = !TEST_CUDA
const TEST_DEVICE_IS_HOST = TEST_OPENMP

const TEST_MPI_ONLY = parse(Bool, get(ENV, "TEST_KOKKOS_MPI_ONLY", "false"))
const TEST_MPI = parse(Bool, get(ENV, "TEST_KOKKOS_MPI", "true")) || TEST_MPI_ONLY

const TEST_BACKEND_HOST          = Kokkos.Serial
const TEST_BACKEND_DEVICE        = TEST_CUDA ? Kokkos.Cuda      : Kokkos.OpenMP
const TEST_UNAVAILABLE_BACKEND   = TEST_CUDA ? Kokkos.HIP       : Kokkos.Cuda
const TEST_UNAVAILABLE_MEM_SPACE = TEST_CUDA ? Kokkos.HIPSpace  : Kokkos.CudaSpace

const TEST_MEM_SPACE_HOST = Kokkos.HostSpace
const TEST_MEM_SPACES_DEVICE = TEST_CUDA ? (Kokkos.CudaSpace, Kokkos.CudaUVMSpace) : (Kokkos.HostSpace,)
const TEST_MAIN_MEM_SPACE_DEVICE = first(TEST_MEM_SPACES_DEVICE)

const TEST_MEM_SHARED = TEST_CUDA ? Kokkos.CudaUVMSpace        : Kokkos.HostSpace
const TEST_MEM_PINNED = TEST_CUDA ? Kokkos.CudaHostPinnedSpace : Kokkos.HostSpace

const TEST_DEVICE_ACCESSIBLE = !TEST_CUDA

const TEST_VIEW_DIMS = (1, 2)
const TEST_VIEW_TYPES = (Float64, Int64)
const TEST_VIEW_LAYOUTS = (Kokkos.LayoutLeft, Kokkos.LayoutRight, Kokkos.LayoutStride)


TEST_CUDA && using CUDA


function print_test_config()
    println("Test configuration:")
    println(" - TEST_OPENMP:           $TEST_OPENMP")
    println(" - TEST_CUDA:             $TEST_CUDA")
    println(" - TEST_MPI:              $TEST_MPI (only MPI: $TEST_MPI_ONLY)")
    println(" - BACKEND_HOST:          $(nameof(TEST_BACKEND_HOST))")
    println(" - BACKEND_DEVICE:        $(nameof(TEST_BACKEND_DEVICE))")
    println(" - UNAVAILABLE_BACKEND:   $(nameof(TEST_UNAVAILABLE_BACKEND))")
    println(" - UNAVAILABLE_MEM_SPACE: $(nameof(TEST_UNAVAILABLE_MEM_SPACE))")
    println(" - MEM_SPACE_HOST:        $(nameof(TEST_MEM_SPACE_HOST))")
    println(" - MEM_SPACES_DEVICE:     $(nameof(TEST_MEM_SPACES_DEVICE))")
    println(" - MAIN_MEM_SPACE_DEVICE: $(nameof(TEST_MAIN_MEM_SPACE_DEVICE))")
    println(" - MEM_SHARED:            $(nameof(TEST_MEM_SHARED))")
    println(" - MEM_PINNED:            $(nameof(TEST_MEM_PINNED))")
    println(" - VIEW_DIMS:             $(TEST_VIEW_DIMS)")
    println(" - VIEW_TYPES:            $(TEST_VIEW_TYPES)")
    println(" - VIEW_LAYOUTS:          $(TEST_VIEW_LAYOUTS)")
end


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
    print_test_config()

    include("pre_wrapper_load.jl")

    Kokkos.build_in_project()  # Use the same directory as Pkg.test uses, forcing a complete compilation

    Kokkos.set_omp_vars()
    @test_logs min_level=Logging.Warn @test_nowarn Kokkos.load_wrapper_lib(; loading_bar=false)
    @test_nowarn Kokkos.initialize()

    if [TEST_BACKEND_HOST, TEST_BACKEND_DEVICE] ⊈ Kokkos.ENABLED_EXEC_SPACES
        error("Invalid execution spaces: $([TEST_BACKEND_HOST, TEST_BACKEND_DEVICE]) ⊈ $(Kokkos.ENABLED_EXEC_SPACES)")
    end

    if !TEST_MPI_ONLY
        include("spaces.jl")

        if TEST_DEVICE_ACCESSIBLE
            include("views.jl")
        else
            include("views_gpu.jl")
        end

        if TEST_CUDA
            include("backends/cuda.jl")
        end

        include("projects.jl")
        include("simple_lib_tests.jl")
    end

    TEST_MPI && include("mpi.jl")

    @info "Tests needed $(length(Kokkos.DynamicCompilation.LOADED_FUNCTION_LIBS)) function libraries"

    !TEST_MPI_ONLY && include("misc.jl")

    GC.gc(true)  # Call the finalizers of all created views
    @test_nowarn Kokkos.finalize()
end
