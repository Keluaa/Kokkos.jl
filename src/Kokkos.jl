module Kokkos

using CxxWrap
using Libdl
using Preferences

export KokkosProject, CMakeKokkosProject, CLibrary
export configure, compile, clean, options, option!
export is_valid, handle, load_lib, unload_lib, is_lib_loaded, get_symbol
export memory_space, execution_space, enabled, main_space_type
export accessible, label


# Configuration options
const KOKKOS_PATH          = @load_preference("kokkos_path")  # Defaults to the packaged installation
const KOKKOS_CMAKE_OPTIONS = @load_preference("cmake_options", [])
const KOKKOS_LIB_OPTIONS   = @load_preference("kokkos_options", Dict())
const KOKKOS_BACKENDS      = @load_preference("backends", ["Serial", "OpenMP"])
const KOKKOS_VIEW_DIMS     = @load_preference("view_dims", [1, 2])
const KOKKOS_VIEW_TYPES    = @load_preference("view_types", ["Float64", "Float32"])
const KOKKOS_BUILD_TYPE    = @load_preference("build_type", "Release")
const KOKKOS_BUILD_DIR     = @load_preference("build_dir", joinpath(pwd(), ".kokkos-build"))


include("dynamic_build.jl")
include("kokkos_lib.jl")
include("utils.jl")

include("kokkos_wrapper.jl")
using .KokkosWrapper

include("spaces.jl")
using .Spaces

include("views.jl")
using .Views


@wrapmodule(KOKKOS_LIB_PATH, :define_kokkos_module)

function __init__()
    @initcxx
end


include("configuration.jl")


"""
    KOKKOS_VERSION::VersionNumber

The Kokkos version currently loaded.
"""
const KOKKOS_VERSION = kokkos_version()

end