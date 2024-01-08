module Kokkos

using Printf
using CxxWrap
using Libdl
using Preferences
using Scratch
using LibGit2
using ProgressMeter

if !isdefined(Base, :get_extension)
    using Requires
end

export KokkosProject, CMakeKokkosProject, CLibrary
export configure, compile, clean, options, option!
export is_valid, handle, load_lib, unload_lib, is_lib_loaded, get_symbol
export memory_space, execution_space, array_layout, enabled, main_space_type
export accessible, label, view_wrap


__get_scratch_build_dir() = joinpath(@get_scratch!("kokkos-build"), string(hash(Base.active_project()), base=16))
__get_tmp_build_dir() = mktempdir(; prefix="jl_kokkos_build_")
__get_project_build_dir(name) = joinpath(dirname(Base.active_project()), name)


# Configuration options
const __DEFAULT_KOKKOS_VERSION_STR   = "4.0.00"  # Must be a valid tag in the Kokkos repo (patch must be 2 digits)
const __DEFAULT_KOKKOS_CMAKE_OPTIONS = []
const __DEFAULT_KOKKOS_LIB_OPTIONS   = []
const __DEFAULT_KOKKOS_BACKENDS      = ["Serial", "OpenMP"]
const __DEFAULT_KOKKOS_BUILD_TYPE    = "Release"
const __DEFAULT_KOKKOS_BUILD_DIR     = __get_scratch_build_dir()

LOCAL_KOKKOS_VERSION_STR = @load_preference("kokkos_version", __DEFAULT_KOKKOS_VERSION_STR)
LOCAL_KOKKOS_DIR = @get_scratch!("kokkos-" * LOCAL_KOKKOS_VERSION_STR)

KOKKOS_PATH          = @load_preference("kokkos_path",    LOCAL_KOKKOS_DIR)
KOKKOS_CMAKE_OPTIONS = @load_preference("cmake_options",  __DEFAULT_KOKKOS_CMAKE_OPTIONS)
KOKKOS_LIB_OPTIONS   = @load_preference("kokkos_options", __DEFAULT_KOKKOS_LIB_OPTIONS)
KOKKOS_BACKENDS      = @load_preference("backends",       __DEFAULT_KOKKOS_BACKENDS)
KOKKOS_BUILD_TYPE    = @load_preference("build_type",     __DEFAULT_KOKKOS_BUILD_TYPE)
KOKKOS_BUILD_DIR     = @load_preference("build_dir",      __DEFAULT_KOKKOS_BUILD_DIR)


"""
    is_kokkos_wrapper_loaded()

Return `true` if [`load_wrapper_lib`](@ref) has been called.
"""
is_kokkos_wrapper_loaded() = !isnothing(Wrapper.KOKKOS_LIB)

function ensure_kokkos_wrapper_loaded()
    if !is_kokkos_wrapper_loaded()
        error("The Kokkos wrapper library is not loaded. \
               Call `Kokkos.initialize` or `Kokkos.load_wrapper_lib`")
    end
end


include("kokkos_project.jl")
include("kokkos_lib.jl")
include("utils.jl")

include("kokkos_wrapper.jl")
using .Wrapper

include("dynamic_compilation.jl")
using .DynamicCompilation

include("spaces.jl")

include("layouts.jl")

include("views.jl")
using .Views

include("configuration.jl")


function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/KokkosAMDGPU.jl")
        @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" include("../ext/KokkosMPI.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/KokkosCUDA.jl")
    end

    Base.atexit(_atexit_hook())
end

end