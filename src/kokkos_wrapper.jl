module Wrapper

using libcxxwrap_julia_jll
using LibGit2
using Libdl
using CxxWrap

import ..Kokkos: CMakeKokkosProject
import ..Kokkos: run_cmd_print_on_error, load_lib, lib_path, build_dir, pretty_compile, clean
import ..Kokkos: ensure_kokkos_wrapper_loaded, configuration_changed!
import ..Kokkos: to_kokkos_version_string, __change_local_version
import ..Kokkos: LOCAL_KOKKOS_DIR, LOCAL_KOKKOS_VERSION_STR
import ..Kokkos: KOKKOS_PATH, KOKKOS_CMAKE_OPTIONS, KOKKOS_LIB_OPTIONS, KOKKOS_BACKENDS
import ..Kokkos: KOKKOS_BUILD_TYPE, KOKKOS_BUILD_DIR

export get_jlcxx_root, get_kokkos_dir, get_kokkos_build_dir, get_kokkos_install_dir
export load_wrapper_lib, get_impl_module


const USE_CLI_GIT = parse(Bool, get(ENV, "JULIA_PKG_USE_CLI_GIT", "false")) || parse(Bool, get(ENV, "JULIA_KOKKOS_USE_CLI_GIT", "false"))


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


function get_latest_kokkos_release(major, minor)
    # Fetch origin to get any new tags
    run_cmd_print_on_error(Cmd(`git fetch`; dir=LOCAL_KOKKOS_DIR))

    if isnothing(major)
        # Get the latest version across all majors
        release_hash = readchomp(Cmd(`git rev-list --tags --max-count=1`; dir=LOCAL_KOKKOS_DIR))
        latest_tag = readchomp(Cmd(`git describe --tags $release_hash`; dir=LOCAL_KOKKOS_DIR))

        if isnothing(tryparse(VersionNumber, latest_tag))
            error("Invalid latest version tag: $latest_tag")
        end
    else
        # Get the latest version matching '<major>.<minor>.*'
        pattern = isnothing(minor) ? "$major.*.*" : "$major.$minor.*"
        all_tags = readchomp(Cmd(`git tag -l $pattern`; dir=LOCAL_KOKKOS_DIR))

        versions = tryparse.(VersionNumber, split(all_tags))
        filter!(!isnothing, versions)
        isempty(versions) && error("No Kokkos release tag matching '$pattern'")

        latest_version = maximum(versions)
        latest_tag = to_kokkos_version_string(latest_version)
    end

    return latest_tag
end


function get_latest_kokkos_release(version_pattern::AbstractString)
    if version_pattern == "latest"
        return get_latest_kokkos_release(nothing, nothing)
    end

    m = match(r"(\d)(?>.(\d))?-latest", version_pattern)
    isnothing(m) && error("Expected version pattern '[<major>[.<minor>]-]latest'")
    return get_latest_kokkos_release(m[1], m[2])
end


function setup_local_kokkos_source()
    KOKKOS_PATH != LOCAL_KOKKOS_DIR && return

    # Clone the Kokkos repo into a scratch directory
    if isempty(readdir(LOCAL_KOKKOS_DIR))
        @info "Cloning Kokkos $LOCAL_KOKKOS_VERSION_STR to $LOCAL_KOKKOS_DIR..."
        @static if USE_CLI_GIT
            run_cmd_print_on_error(Cmd(`git clone https://github.com/kokkos/kokkos.git .`; dir=LOCAL_KOKKOS_DIR))
            repo = nothing
        else
            repo = LibGit2.clone("https://github.com/kokkos/kokkos.git", LOCAL_KOKKOS_DIR)
        end
    else
        repo = LibGit2.GitRepo(LOCAL_KOKKOS_DIR)
    end

    if endswith(LOCAL_KOKKOS_VERSION_STR, "latest")
        new_local_version = get_latest_kokkos_release(LOCAL_KOKKOS_VERSION_STR)
        @debug "Kokkos version '$LOCAL_KOKKOS_VERSION_STR' was resolved to '$new_local_version'"
        __change_local_version(new_local_version)  # Updates LOCAL_KOKKOS_VERSION_STR, etc...
    elseif isnothing(tryparse(VersionNumber, LOCAL_KOKKOS_VERSION_STR))
        error("Invalid Kokkos version tag: $LOCAL_KOKKOS_VERSION_STR")
    end

    # Get the commit hash for the version tag
    @static if USE_CLI_GIT
        tag_str = "tags/$LOCAL_KOKKOS_VERSION_STR"
        release_hash = readchomp(Cmd(`git rev-list -n 1 $tag_str`; dir=LOCAL_KOKKOS_DIR))
    else
        release_commit = LibGit2.GitObject(repo, LOCAL_KOKKOS_VERSION_STR)
        release_hash = string(LibGit2.GitHash(release_commit))
    end

    @debug "Checkout Kokkos $LOCAL_KOKKOS_VERSION_STR (hash: $release_hash) in repo at $LOCAL_KOKKOS_DIR..."
    @static if USE_CLI_GIT
        run_cmd_print_on_error(Cmd(`git checkout -q $release_hash`; dir=LOCAL_KOKKOS_DIR))
    else
        LibGit2.checkout!(repo, release_hash)
    end
end


function create_kokkos_lib_project(; no_git=false)
    !no_git && setup_local_kokkos_source()

    julia_exe_path = joinpath(Sys.BINDIR, "julia")
    !isfile(julia_exe_path) && error("Could not determine the position of the Julia executable")

    jlcxx_root = get_jlcxx_root()

    build_dir = joinpath(KOKKOS_BUILD_DIR, "wrapper-build-$(lowercase(KOKKOS_BUILD_TYPE))")
    install_dir = joinpath(build_dir, "install")
    kokkos_path = @something KOKKOS_PATH get_kokkos_dir()

    cmake_options = [
        "-DCMAKE_INSTALL_PREFIX=$install_dir",
        # Kokkos build options
        # Required: else Kokkos is statically linked, as well as any library built
        # afterwards. This will create a nasty invisible problem where a process could have several
        # Kokkos independent runtimes: therefore shared libs is a must.
        "-DBUILD_SHARED_LIBS=ON",
        # CxxWrap options
        "-DJulia_EXECUTABLE=$julia_exe_path",
        "-DJlCxx_ROOT=$jlcxx_root"
    ]
    append!(cmake_options, KOKKOS_CMAKE_OPTIONS)

    kokkos_options = Dict{String, String}()

    enabled_backends = uppercase.(KOKKOS_BACKENDS)
    for backend in enabled_backends
        kokkos_options["Kokkos_ENABLE_" * backend] = "ON"
    end

    # Disable all other Kokkos backends, to make sure no cached variable keeps one of them enabled
    all_backends = parentmodule(@__MODULE__).ALL_BACKENDS .|> nameof .|> string .|> uppercase
    for backend in setdiff(all_backends, enabled_backends)
        kokkos_options["Kokkos_ENABLE_" * backend] = "OFF"
    end

    for option in KOKKOS_LIB_OPTIONS
        name, val = rsplit(option, '='; limit=2)
        kokkos_options[name] = val
    end

    return CMakeKokkosProject(joinpath(@__DIR__, "../lib/kokkos_wrapper"), "libKokkosWrapper";
        target = "KokkosWrapper",
        build_dir, build_type = KOKKOS_BUILD_TYPE,
        cmake_options, kokkos_path, kokkos_options,
        inherit_options = false
    )
end


function install_kokkos()
    @debug "Installing Kokkos to $(get_kokkos_install_dir())"
    run_cmd_print_on_error(Cmd(`cmake --build $(build_dir(KOKKOS_LIB_PROJECT)) --target install`;
        dir=KOKKOS_LIB_PROJECT.commands_dir))
end


function clear_compilation_lock end  # Defined in DynamicCompilation


function compile_wrapper_lib(; no_compilation=false, no_git=false, loading_bar=true)
    global KOKKOS_LIB_PROJECT = create_kokkos_lib_project(; no_git)
    global KOKKOS_LIB_PATH = lib_path(KOKKOS_LIB_PROJECT)

    if no_compilation
        configuration_changed!(KOKKOS_LIB_PROJECT, false)
        return
    end

    pretty_compile(KOKKOS_LIB_PROJECT; info=true, loading_bar)
    install_kokkos()
    mkpath(joinpath(build_dir(KOKKOS_LIB_PROJECT), "func_libs"))
    clear_compilation_lock()
end


KOKKOS_LIB_PROJECT = nothing
KOKKOS_LIB_PATH = nothing
KOKKOS_LIB = nothing


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


function get_kokkos_func_libs_dir()
    ensure_kokkos_wrapper_loaded()
    return joinpath(build_dir(KOKKOS_LIB_PROJECT), "func_libs")
end


"""
    clean_cmake_files()

Clears the CMake cache of the wrapper library, and removes all build files, as well as compiled
function libraries.
"""
function clean_cmake_files()
    isnothing(KOKKOS_LIB_PROJECT) && error("the CMake project of the wrapper library is not set up")
    clean(KOKKOS_LIB_PROJECT; reset=true)
end


module Impl
    using CxxWrap
end


get_impl_module() = Impl


"""
    load_wrapper_lib(; no_compilation=false, no_git=false, loading_bar=true)

Configures, compiles then loads the wrapper library using the current [Configuration Options](@ref).

After calling this method, all configuration options become locked.

If `no_compilation` is `true`, then the CMake project of the wrapper library will not be configured
or compiled.

If `no_git` is `true`, then if we need to use the Kokkos installation of the package, no Git
operations (clone + checkout) will be done.

Both `no_compilation=true` and `no_git=true` are needed when initializing Kokkos.jl in non-root MPI
processes.
"""
function load_wrapper_lib(; no_compilation=false, no_git=false, loading_bar=true)
    !isnothing(KOKKOS_LIB) && return

    compile_wrapper_lib(; no_compilation, no_git, loading_bar)

    @debug "Loading the Kokkos Wrapper library..."

    # RTLD_GLOBAL is needed to make Kokkos symbols available to all libraries
    global KOKKOS_LIB = load_lib(KOKKOS_LIB_PROJECT; flags=Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)

    # This seemingly innocent macro call will add methods in all modules.
    # All of those methods will be imported into the `Impl` module then overloaded here.
    # While each of them are defined in their respective module, all specializations are stored in
    # `Impl`.
    # Methods meant to define variables are prefixed by '__' and defined only in `Impl`. Then they
    # are called through '__init_vars()', circumventing the requirement that variables can only be
    # set by methods of the same module.
    Core.eval(Impl, quote
        @static if pkgversion(CxxWrap) >= v"0.14-"
            @wrapmodule(() -> $KOKKOS_LIB_PATH, :define_kokkos_module)
        else
            @wrapmodule($KOKKOS_LIB_PATH, :define_kokkos_module)
        end
    end)

    Kokkos = parentmodule(Wrapper)
    Kokkos.__init_vars()
    Kokkos.__init_spaces_vars()

    return
end

end
