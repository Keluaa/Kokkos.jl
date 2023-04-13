
HAS_CONFIGURATION_CHANGED = false


function set_kokkos_version(version::Union{Nothing, Missing, String})
    @set_preferences!("kokkos_version" => version)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New Kokkos version set to '$version'.\n\
           Restart your Julia session for this change to take effect.\n\
           Note that the version is only used if 'kokkos_path' is not set."
end


function set_kokkos_path(path::Union{Nothing, Missing, String})
    @set_preferences!("kokkos_path" => path)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New Kokkos path set to '$path'.\n\
           Restart your Julia session for this change to take effect."
end


function set_cmake_options(options::Union{Nothing, Missing, Vector{String}})
    @set_preferences!("cmake_options" => options)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New Kokkos wrapper CMake options set to $options.\n\
           Restart your Julia session for this change to take effect."
end


function set_kokkos_options(options::Union{Nothing, Missing, Vector{String}})
    @set_preferences!("kokkos_options" => options)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New Kokkos options set to $options.\n\
           Restart your Julia session for this change to take effect."
end


function set_backends(backends::Union{Nothing, Missing, Vector{String}})
    @set_preferences!("backends" => backends)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New backends set to $backends.\n\
           Restart your Julia session for this change to take effect."
end

set_backends(backends::Vector{<:Kokkos.ExecutionSpace}) = set_backends(string.(nameof.(backends)))


function set_view_dims(dims::Union{Nothing, Missing, Vector{Int}})
    @set_preferences!("view_dims" => dims)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New view dimensions set to $dims.\n\
           Restart your Julia session for this change to take effect."
end


function set_view_types(types::Union{Nothing, Missing, Vector{String}})
    @set_preferences!("view_types" => types)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New view types set to $types.\n\
           Restart your Julia session for this change to take effect."
end

set_view_types(types::Vector{Type}) = set_view_types(string.(nameof.(types)))


function set_build_type(build_type::Union{Nothing, Missing, String})
    @set_preferences!("build_type" => build_type)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New build type set to $build_type.\n\
           Restart your Julia session for this change to take effect."
end


function set_build_dir(build_dir::Union{Nothing, Missing, String})
    @set_preferences!("build_dir" => build_dir)
    global HAS_CONFIGURATION_CHANGED = true
    @info "New build directory set to $build_dir.\n\
           Restart your Julia session for this change to take effect."
end