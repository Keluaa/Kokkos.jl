
function to_kokkos_version_string(version::VersionNumber)
    return @sprintf("%d.%d.%02d", version.major, version.minor, version.patch)
end


"""
    set_omp_vars(;
        places = "cores",
        bind = "close",
        num_threads = Threads.nthreads()
    )

Helper function to set the main OpenMP environment variables used by Kokkos. It must be called
before calling [`initialize`](@ref).

`places` sets `OMP_PLACES`.
`bind` sets `OMP_PROC_BIND`.
`num_threads` sets `OMP_NUM_THREADS`.

Note that since Julia threads and OpenMP threads are decoupled, there is no constraint imposed by
Julia on OpenMP threads: there can be as many threads as needed.

!!! warning

    Pinning the Julia threads with [ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl)
    or with the `JULIA_EXCLUSIVE` environment variable can have an impact on OpenMP thread
    affinities, making the OpenMP variables useless.
"""
function set_omp_vars(; places = "cores", bind = "close", num_threads = Base.Threads.nthreads())
    if is_initialized()
        error("OpenMP variables should be set before initializing Kokkos")
    end

    ENV["OMP_PLACES"] = places
    ENV["OMP_PROC_BIND"] = bind
    ENV["OMP_NUM_THREADS"] = num_threads

    julia_exclusive = get(ENV, "JULIA_EXCLUSIVE", "0")
    if !isempty(julia_exclusive) && julia_exclusive != "0"
        @warn "Environment variable 'JULIA_EXCLUSIVE' is set, which can affect OpenMP thread affinities"
    end
end


function warn_config_changed()
    if HAS_CONFIGURATION_CHANGED
        @warn "Kokkos configuration changed! Restart Julia for those changes to take effect!" maxlog=1
    end
end


function new_initialization_settings()
    ensure_kokkos_wrapper_loaded()
    return Base.invokelatest(get_impl_module().InitializationSettings)
end


# Initialization settings getters, defined in 'kokkos_wrapper.cpp', in 'define_initialization_settings'
function num_threads end
function device_id end
function disable_warnings end
function print_configuration end
function tune_internals end
function tools_libs end
function tools_args end
function map_device_id_by end


# Initialization settings setters, defined in 'kokkos_wrapper.cpp', in 'define_initialization_settings'
function num_threads! end
function device_id! end
function disable_warnings! end
function print_configuration! end
function tune_internals! end
function tools_libs! end
function tools_args! end
function map_device_id_by! end


"""
    initialize(;
        num_threads=nothing,
        device_id=nothing, map_device_id_by=nothing,
        disable_warnings=nothing, print_configuration=nothing,
        tune_internals=nothing,
        tools_libs=nothing, tools_args=nothing
    )

Initializes Kokkos by calling [`Kokkos::initialize()`](https://kokkos.github.io/kokkos-core-wiki/API/core/initialize_finalize/initialize.html).

The keyword arguments build are passed to the
[`InitializationSettings`](https://kokkos.github.io/kokkos-core-wiki/API/core/initialize_finalize/InitializationSettings.html)
constructor which is then passed to `Kokkos::initialize()`. A value of `nothing` keeps the default
behaviour of Kokkos.

The Kokkos wrapper library is loaded (and recompiled if needed) if it is not already the case.
This locks the current [Configuration Options](@ref) until the end of the current Julia session.
"""
function initialize(;
    num_threads=nothing,
    device_id=nothing, map_device_id_by=nothing,
    disable_warnings=nothing, print_configuration=nothing,
    tune_internals=nothing,
    tools_libs=nothing, tools_args=nothing
)
    warn_config_changed()
    load_wrapper_lib()
    settings = new_initialization_settings()
    !isnothing(num_threads)         && num_threads!(settings, num_threads)
    !isnothing(device_id)           && device_id!(settings, device_id)
    !isnothing(disable_warnings)    && disable_warnings!(settings, disable_warnings)
    !isnothing(print_configuration) && print_configuration!(settings, print_configuration)
    !isnothing(tune_internals)      && tune_internals!(settings, tune_internals)
    !isnothing(tools_libs)          && tools_libs!(settings, tools_libs)
    !isnothing(tools_args)          && tools_args!(settings, tools_args)
    !isnothing(map_device_id_by)    && map_device_id_by!(settings, map_device_id_by)
    Base.invokelatest(initialize, settings)
end


# Defined in 'kokkos_wrapper.cpp', in 'define_kokkos_module'
"""
    print_configuration(io::IO, verbose::Bool)

Prints the internal Kokkos configuration to `io`.

Equivalent to `Kokkos::print_configuration(out, verbose)`.
"""
function print_configuration end


# Defined in 'kokkos_wrapper.cpp', in 'define_kokkos_module'
"""
    finalize()

Calls [`Kokkos::finalize()`](https://kokkos.github.io/kokkos-core-wiki/API/core/initialize_finalize/finalize.html).

!!! info

    If Kokkos isn't already finalized, `finalize` will be called automatically at process exit
    through `Base.atexit`.

!!! warning

    Kokkos requires that all view destructors should be called **before** calling `finalize`.
    This is done automatically for all views allocated through `Kokkos.jl` upon calling `finalize`,
    and therefore they will all become invalid.
"""
function finalize end


"""
    is_initialized()

Return `Kokkos::is_initialized()`.

Can be called before the wrapper library is loaded.
"""
function is_initialized()
    !is_kokkos_wrapper_loaded() && return false
    # Defined in 'kokkos_wrapper.cpp', in 'define_kokkos_module'
    return Kokkos.Wrapper.Impl.is_initialized()
end


"""
    is_finalized()

Return `Kokkos::is_finalized()`.

Can be called before the wrapper library is loaded.
"""
function is_finalized()
    !is_kokkos_wrapper_loaded() && return false
    # Defined in 'kokkos_wrapper.cpp', in 'define_kokkos_module'
    return Kokkos.Wrapper.Impl.is_finalized()
end


function _atexit_hook()
    !is_kokkos_wrapper_loaded() && return
    !is_initialized() && return  # Either `Kokkos.initialize` was never called, or `Kokkos.finalize` was already called
    is_finalized() && return  # `Kokkos.finalize` was already called
    Kokkos.finalize()
end


# Defined in 'kokkos_wrapper.cpp', in 'define_kokkos_module'
"""
    fence()
    fence(label::String)

Wait for all asynchronous Kokkos operations to complete.

Equivalent to [`Kokkos::fence()`](https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/fence.html).
"""
function fence end


"""
    configinfo(io::IO = stdout)

Print information about all configuration options.
"""
function configinfo(io::IO = stdout)
    if HAS_CONFIGURATION_CHANGED
        println(io, "Configuration changed! \
                     Those values are not the ones the loaded Kokkos library is using.\n")
    end

    print(io, "Kokkos path: '", KOKKOS_PATH, "'")
    if KOKKOS_PATH == LOCAL_KOKKOS_DIR
        println(io, " (installation of Kokkos.jl)")
        println(io, "Kokkos version: ", LOCAL_KOKKOS_VERSION_STR)
    else
        println(io)
    end

    println(io, "CMake options: ", `$KOKKOS_CMAKE_OPTIONS`)
    println(io, "CMake build type: ", KOKKOS_BUILD_TYPE)
    println(io, "CMake build dir: '", KOKKOS_BUILD_DIR, "'")
    println(io, "Kokkos options: ", `$KOKKOS_LIB_OPTIONS`)
    println(io, "Enabled Kokkos backends: ", join(KOKKOS_BACKENDS, ", "))
end


"""
    versioninfo(io::IO = stdout; internal=true, verbose=false)

Print the version and various information about the underlying Kokkos library.

If `internal` is true, `Kokkos::print_configuration()` is called. `verbose` is passed to that call.

This function does not require for Kokkos to be initialized, however if `internal=true` then the
output will be incomplete.
"""
function versioninfo(io::IO = stdout; internal=true, verbose=false)
    ensure_kokkos_wrapper_loaded()
    warn_config_changed()
    print(io, "Kokkos version $KOKKOS_VERSION")
    if isnothing(KOKKOS_PATH)
        println(io, " (packaged sources)")
    else
        println(io, " (path: $(KOKKOS_PATH))")
    end
    println(io, "Kokkos installation dir: ", get_kokkos_install_dir())
    println(io, "Wrapper library compiled at ", build_dir(Wrapper.KOKKOS_LIB_PROJECT))
    println(io, "\nEnabled execution spaces:")
    for space in ENABLED_EXEC_SPACES
        println(io, " - $(nameof(space)) (default memory space: $(nameof(memory_space(space))))")
    end
    println(io, "\nEnabled memory spaces:")
    for space in ENABLED_MEM_SPACES
        println(io, " - $(nameof(space)) (associated execution space: $(nameof(execution_space(space))))")
    end

    if internal
        println(io, "\nKokkos internal configuration:")
        print_configuration(io, verbose)
    end
end


"""
    KOKKOS_VERSION::VersionNumber

The Kokkos version currently loaded.

`nothing` if Kokkos is not yet loaded.
See [kokkos_version](@ref) for the version of the packaged installation of Kokkos, which
is defined before loading Kokkos.
"""
KOKKOS_VERSION = nothing


function __change_local_version(new_local_version)
    if is_kokkos_wrapper_loaded()
        error("Cannot update local Kokkos version variables after the wrapped was loaded")
    end
    global LOCAL_KOKKOS_VERSION_STR = String(new_local_version)
end


function __validate_parameters(;
    view_layout, view_dim, view_type,
    exec_space, mem_space,
    dest_layout, dest_space,
    subview_dim
)
    all_dims = filter(!isnothing, union([view_dim], [subview_dim]))
    if !all(d -> (0 ≤ d ≤ 8), all_dims)
        wrong_dims = filter(d -> (0 ≤ d ≤ 8), all_dims)
        wrong_dims_str = join(wrong_dims, ", ", " and ")
        error("Kokkos only supports dimensions from 0 to 8, got: " * wrong_dims_str)
    end

    if !isnothing(exec_space) && !enabled(main_space_type(exec_space))
        error("Cannot compile for disabled execution space: " * main_space_type(exec_space))
    end

    all_mem_spaces = filter(!isnothing, union([mem_space], [dest_space]))
    if any((!enabled ∘ main_space_type).(all_mem_spaces))
        wrong_mem = filter(!enabled ∘ main_space_type, all_mem_spaces)
        wrong_mem_str = join(wrong_mem, ", ", " and ")
        error("Cannot compile for disabled memory space: " * wrong_mem_str)
    end

    # Convert to string
    view_dim    = isnothing(view_dim)    ? "" : string(view_dim)
    subview_dim = isnothing(subview_dim) ? "" : string(subview_dim)
    view_type   = isnothing(view_type)   ? "" : Wrapper.julia_type_to_c(view_type)
    exec_space  = isnothing(exec_space)  ? "" : string(nameof(main_space_type(exec_space)))
    mem_space   = isnothing(mem_space)   ? "" : string(nameof(main_space_type(mem_space)))
    dest_space  = isnothing(dest_space)  ? "" : string(nameof(main_space_type(dest_space)))
    view_layout = isnothing(view_layout) ? "" : lowercase(string(nameof(view_layout)))[7:end]  # Remove leading 'Layout'
    dest_layout = isnothing(dest_layout) ? "" : lowercase(string(nameof(dest_layout)))[7:end]

    return view_layout, view_dim, view_type,
           exec_space, mem_space,
           dest_layout, dest_space,
           subview_dim
end


function __init_vars()
    impl = get_impl_module()
    global KOKKOS_VERSION = Base.invokelatest(impl.__kokkos_version)
end
