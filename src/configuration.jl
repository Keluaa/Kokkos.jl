
HAS_CONFIGURATION_CHANGED = false


function set_kokkos_version(version::Union{Nothing, Missing, String})
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


function set_kokkos_path(path::Union{Nothing, Missing, String})
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


function set_cmake_options(options::Union{Nothing, Missing, Vector{String}})
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


function set_kokkos_options(options::Union{Nothing, Missing, Vector{String}})
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


function set_backends(backends::Union{Nothing, Missing, Vector{String}})
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


function set_view_dims(dims::Union{Nothing, Missing, Vector{Int}})
    @set_preferences!("view_dims" => dims)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_VIEW_DIMS = @load_preference("view_dims", __DEFAULT_KOKKOS_VIEW_DIMS)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New view dimensions set to $dims.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_VIEW_DIMS
end


function set_view_types(types::Union{Nothing, Missing, Vector{String}})
    @set_preferences!("view_types" => types)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_VIEW_TYPES = @load_preference("view_types", __DEFAULT_KOKKOS_VIEW_TYPES)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New view types set to $types.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_VIEW_TYPES
end

set_view_types(types::Vector{DataType}) = set_view_types(string.(nameof.(types)))


function set_view_layouts(layouts::Union{Nothing, Missing, Vector{String}})
    @set_preferences!("view_layouts" => layouts)
    if !is_kokkos_wrapper_loaded()
        global KOKKOS_VIEW_LAYOUTS = @load_preference("view_layouts", __DEFAUlT_KOKKOS_VIEW_LAYOUTS)
    else
        global HAS_CONFIGURATION_CHANGED = true
        @info "New view layouts set to $layouts.\n\
               Restart your Julia session for this change to take effect."
    end
    return KOKKOS_VIEW_LAYOUTS
end

function set_view_layouts(layouts::Vector{DataType})
    typeassert.(layouts, Type{<:Kokkos.Layout})
    # Transform the types to the names as described in the docs
    layouts_str = layouts .|> nameof .|> string .|> lowercase
    layouts_str = chop.(layouts_str; head=length("Layout"), tail=0)  # Remove the leading 'Layout'
    return set_view_layouts(string.(layouts_str))
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


function set_build_dir(build_dir::Union{Nothing, Missing, String})
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
