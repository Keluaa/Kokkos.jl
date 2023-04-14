module KokkosWrapper

using libcxxwrap_julia_jll
using LibGit2
using CxxWrap

import ..Kokkos: CMakeKokkosProject
import ..Kokkos: run_cmd_print_on_error, load_lib, lib_path, build_dir, pretty_compile
import ..Kokkos: ensure_kokkos_wrapper_loaded
import ..Kokkos: LOCAL_KOKKOS_DIR, LOCAL_KOKKOS_VERSION_STR
import ..Kokkos: KOKKOS_PATH, KOKKOS_CMAKE_OPTIONS, KOKKOS_LIB_OPTIONS, KOKKOS_BACKENDS
import ..Kokkos: KOKKOS_VIEW_DIMS, KOKKOS_VIEW_TYPES, KOKKOS_BUILD_TYPE, KOKKOS_BUILD_DIR

export get_jlcxx_root, get_kokkos_dir, get_kokkos_build_dir, get_kokkos_install_dir
export load_wrapper_lib, get_impl_module


"""
    get_jlcxx_root()

Return the directory where the file "JlCxxConfig.cmake" is located for the currently loaded CxxWrap
package.

Setting the CMake variable `JlCxx_ROOT` to this path allows the CMake function `find_package` to
load JlCxx. 
"""
get_jlcxx_root() = libcxxwrap_julia_jll.artifact_dir


"""
    get_kokkos_dir()

The directory where the sources of Kokkos are located.

If `KOKKOS_PATH` is not set, it defaults to the sources of Kokkos packaged with `Kokkos.jl`.

This directory is meant to be passed to the CMake function `add_subdirectory` in order to load
Kokkos as an in-tree build.
"""
get_kokkos_dir() = KOKKOS_PATH


julia_type_to_c(::Type{Float64}) = "double"
julia_type_to_c(::Type{Float32}) = "float"
julia_type_to_c(::Type{Float16}) = "_Float16"  # Not exactly standard, but in C++23 we will have std::float16_t through #include <stdfloat>
julia_type_to_c(::Type{Int64})   = "int64_t"
julia_type_to_c(::Type{Int32})   = "int32_t"
julia_type_to_c(::Type{Int16})   = "int16_t"
julia_type_to_c(::Type{Int8})    = "int8_t"
julia_type_to_c(::Type{UInt64})  = "uint64_t"
julia_type_to_c(::Type{UInt32})  = "uint32_t"
julia_type_to_c(::Type{UInt16})  = "uint16_t"
julia_type_to_c(::Type{UInt8})   = "uint8_t"
julia_type_to_c(::Type{Bool})    = "bool"
julia_type_to_c(::Type)          = error("no known equivalent scalar C type for $t")


function julia_str_type_to_c_type(t::String)
    julia_type = try
        getfield(Main, Symbol(t))
    catch
        error("unknown Julia type: $t")
    end
    return julia_type_to_c(julia_type)
end


function setup_local_kokkos_source()
    KOKKOS_PATH != LOCAL_KOKKOS_DIR && return
    # Download our local Kokkos source files into a scratch directory
    if isempty(readdir(LOCAL_KOKKOS_DIR))
        @info "Cloning Kokkos $LOCAL_KOKKOS_VERSION_STR to $LOCAL_KOKKOS_DIR..."
        repo = LibGit2.clone("https://github.com/kokkos/kokkos.git", LOCAL_KOKKOS_DIR)
    else
        repo = LibGit2.GitRepo(LOCAL_KOKKOS_DIR)
    end
    release_commit = LibGit2.GitObject(repo, LOCAL_KOKKOS_VERSION_STR)
    release_hash = string(LibGit2.GitHash(release_commit))
    @debug "Checkout Kokkos $LOCAL_KOKKOS_VERSION_STR (hash: $release_hash) in repo at $LOCAL_KOKKOS_DIR..."
    LibGit2.checkout!(repo, release_hash)
end


function create_kokkos_lib_project()
    setup_local_kokkos_source()

    julia_exe_path = joinpath(Sys.BINDIR, "julia")
    !isfile(julia_exe_path) && error("Could not determine the position of the Julia executable")

    c_view_types = julia_str_type_to_c_type.(KOKKOS_VIEW_TYPES)

    jlcxx_root = get_jlcxx_root()

    build_dir = joinpath(KOKKOS_BUILD_DIR, "wrapper-build-$(lowercase(KOKKOS_BUILD_TYPE))")
    install_dir = joinpath(build_dir, "install")
    kokkos_path = @something KOKKOS_PATH get_kokkos_dir()

    cmake_options = [
        "-DCMAKE_INSTALL_PREFIX=$install_dir",
        "-DJulia_EXECUTABLE=$julia_exe_path",
        "-DJlCxx_ROOT=$jlcxx_root",
        "-DVIEW_DIMENSIONS='" * join(KOKKOS_VIEW_DIMS, ",") * "'",
        "-DVIEW_TYPES='" * join(c_view_types, ",") * "'"
    ]
    append!(cmake_options, KOKKOS_CMAKE_OPTIONS)

    kokkos_options = Dict{String, String}()
    for backend in KOKKOS_BACKENDS
        kokkos_options["Kokkos_ENABLE_" * uppercase(backend)] = "ON"
    end

    for option in KOKKOS_LIB_OPTIONS
        name, val = rsplit(option, '='; limit=2)
        kokkos_options[name] = val
    end

    return CMakeKokkosProject(joinpath(@__DIR__, "../lib/kokkos_wrapper"), "libKokkosWrapper";
        build_dir, build_type = KOKKOS_BUILD_TYPE, cmake_options, kokkos_path, kokkos_options
    )
end


function install_kokkos()
    @debug "Installing Kokkos to $(get_kokkos_install_dir())"
    run_cmd_print_on_error(Cmd(`cmake --build $(build_dir(KOKKOS_LIB_PROJECT)) --target install`;
        dir=KOKKOS_LIB_PROJECT.commands_dir))
end


function compile_wrapper_lib()
    global KOKKOS_LIB_PROJECT = create_kokkos_lib_project()
    global KOKKOS_LIB_PATH = lib_path(KOKKOS_LIB_PROJECT)

    pretty_compile(KOKKOS_LIB_PROJECT; info=true)
    install_kokkos()
end


KOKKOS_LIB_PROJECT = nothing
KOKKOS_LIB_PATH = nothing
KOKKOS_LIB = nothing


is_kokkos_wrapper_compiled() = !isnothing(KOKKOS_LIB_PATH)


"""
    get_kokkos_build_dir()

The directory where Kokkos is compiled.
"""
function get_kokkos_build_dir()
    ensure_kokkos_wrapper_loaded()
    return joinpath(build_dir(KOKKOS_LIB_PROJECT), "lib", "kokkos")
end


"""
    get_kokkos_install_dir()

The directory where Kokkos is installed.

This directory can be passed to the CMake function `find_package` through the `Kokkos_ROOT` variable
in order to load Kokkos with the same options and backends as the ones used by `Kokkos.jl`.
"""
function get_kokkos_install_dir()
    ensure_kokkos_wrapper_loaded()
    return joinpath(build_dir(KOKKOS_LIB_PROJECT), "install")
end


module Impl
    using CxxWrap
end


get_impl_module() = Impl


"""
    load_wrapper_lib()

Configures, compiles then loads the wrapper library using the current [Configuration Options](@ref).

After calling this method, all configuration options become locked.
"""
function load_wrapper_lib()
    !isnothing(KOKKOS_LIB) && return
    !is_kokkos_wrapper_compiled() && compile_wrapper_lib()

    @debug "Loading the Kokkos Wrapper library..."

    global KOKKOS_LIB = load_lib(KOKKOS_LIB_PROJECT)

    # This seemingly innocent macro call will add methods in all modules.
    # All of those methods will be imported into the `Impl` module then overloaded here.
    # While each of them are defined in their respective module, all specializations are stored in
    # `Impl`.
    # Methods meant to define variables are prefixed by '__' and defined only in `Impl`. Then they
    # are called through '__init_vars()', circumventing the requirement that variables can only be
    # set by methods of the same module.
    Core.eval(Impl, quote
        @wrapmodule($KOKKOS_LIB_PATH, :define_kokkos_module)
    end)

    Kokkos = parentmodule(KokkosWrapper)
    Kokkos.__init_vars()
    Kokkos.Spaces.__init_vars()
    Kokkos.Views.__init_vars()

    return
end

end