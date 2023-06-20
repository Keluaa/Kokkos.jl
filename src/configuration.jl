
HAS_CONFIGURATION_CHANGED = false


function set_kokkos_version(version::Union{Nothing, Missing, AbstractString})
    if version isa String && findlast('.', version) != length(version) - 2
        error("Invalid Kokkos version. The version must be a valid tag in the Kokkos repo: \
               the patch version must have two digits. Use '3.7.02' and not '3.7.2'.")
    end
    @set_preferences!("kokkos_version" => version)
    if !is_kokkos_wrapper_loaded()
        global LOCAL_KOKKOS_VERSION_STR = @load_preference("kokkos_version", __DEFAULT_KOKKOS_VERSION_STR)
        global LOCAL_KOKKOS_DIR = @get_scratch!("kokkos-" * LOCAL_KOKKOS_VERSION_STR)
        global KOKKOS_PATH = @load_preference("kokkos_path", LOCAL_KOKKOS_DIR)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New Kokkos version set to '$version'.\n\
               Restart your Julia session for this change to take effect.\n\
               Note that the version is only used if 'kokkos_path' is not set."
    end
    return LOCAL_KOKKOS_VERSION_STR
end


function set_kokkos_version(version::VersionNumber)
    return set_kokkos_version(@sprintf("%d.%d.%02d", version.major, version.minor, version.patch))
end


function set_kokkos_path(path::Union{Nothing, Missing, AbstractString})
    @set_preferences!("kokkos_path" => path)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_PATH = @load_preference("kokkos_path", LOCAL_KOKKOS_DIR)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New Kokkos path set to '$path'.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_PATH
end


function set_cmake_options(options::Union{Nothing, Missing, Vector{<:AbstractString}})
    @set_preferences!("cmake_options" => options)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_CMAKE_OPTIONS = @load_preference("cmake_options", __DEFAULT_KOKKOS_CMAKE_OPTIONS)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New Kokkos wrapper CMake options set to $options.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_CMAKE_OPTIONS
end


function set_kokkos_options(options::Union{Nothing, Missing, Vector{<:AbstractString}})
    @set_preferences!("kokkos_options" => options)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_LIB_OPTIONS = @load_preference("kokkos_options", __DEFAULT_KOKKOS_LIB_OPTIONS)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New Kokkos options set to $options.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_LIB_OPTIONS
end

function set_kokkos_options(options::Dict{String, <:Any})
    options_list = String[]
    for (name, val) in options
        if val isa Bool
            val = val ? "ON" : "OFF"
        end
        push!(options_list, "$(name)=$(val)")
    end
    return set_kokkos_options(options_list)
end


function set_backends(backends::Union{Nothing, Missing, Vector{<:AbstractString}})
    @set_preferences!("backends" => backends)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_BACKENDS = @load_preference("backends", __DEFAULT_KOKKOS_BACKENDS)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New backends set to $backends.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_BACKENDS
end

function set_backends(backends::Vector{DataType})
    typeassert.(backends, Type{<:Kokkos.ExecutionSpace})
    return set_backends(string.(nameof.(backends)))
end


function set_build_type(build_type::Union{Nothing, Missing, String})
    @set_preferences!("build_type" => build_type)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_BUILD_TYPE = @load_preference("build_type", __DEFAULT_KOKKOS_BUILD_TYPE)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New build type set to $build_type.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_BUILD_TYPE
end


function set_build_dir(build_dir::Union{Nothing, Missing, AbstractString}; local_only=false)
    if local_only
        # KOKKOS_BUILD_DIR is the only option which should be set on all processes in a MPI app
        global KOKKOS_BUILD_DIR = build_dir
        return KOKKOS_BUILD_DIR
    end

    @set_preferences!("build_dir" => build_dir)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_BUILD_DIR = @load_preference("build_dir", __DEFAULT_KOKKOS_BUILD_DIR)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New build directory set to $build_dir.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_BUILD_DIR
end


"""
    build_in_scratch()

Sets the `build_dir` configuration option to a scratch directory (the default).
"""
build_in_scratch() = set_build_dir(__get_scratch_build_dir())


"""
    build_in_tmp()

Sets the `build_dir` configuration option to a temporary directory, cleaned at the end of the Julia
session.
"""
build_in_tmp() = set_build_dir(__get_tmp_build_dir())


"""
    build_in_project(name = ".kokkos-build")

Sets the `build_dir` configuration option to the directory `name` in the active project directory.
"""
build_in_project(name = ".kokkos-build") = set_build_dir(__get_project_build_dir(name))
