
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

!!! warning

    Kokkos requires that all view destructors should be called **before** calling `finalize`.
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
    return Kokkos.KokkosWrapper.Impl.is_initialized()
end


"""
    is_finalized()

Return `Kokkos::is_finalized()`.

Can be called before the wrapper library is loaded.
"""
function is_finalized()
    !is_kokkos_wrapper_loaded() && return false
    # Defined in 'kokkos_wrapper.cpp', in 'define_kokkos_module'
    return Kokkos.KokkosWrapper.Impl.is_finalized()
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
    println(io, "Views:")
    println(io, " - dimensions: ", join(KOKKOS_VIEW_DIMS, ", "))
    println(io, " - types: ", join(KOKKOS_VIEW_TYPES, ", "))
    println(io, " - layouts: ", join(KOKKOS_VIEW_LAYOUTS, ", "))
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
    println(io, "Wrapper library compiled at ", build_dir(KokkosWrapper.KOKKOS_LIB_PROJECT))
    println(io, "\nCompiled execution spaces:")
    for space in COMPILED_EXEC_SPACES
        println(io, " - $(nameof(space)) (default memory space: $(nameof(memory_space(space))))")
    end
    println(io, "\nCompiled memory spaces:")
    for space in COMPILED_MEM_SPACES
        println(io, " - $(nameof(space)) (associated execution space: $(nameof(execution_space(space))))")
    end
    println(io, "\nCompiled view options:")
    println(io, " - types:      ", join(COMPILED_TYPES, ", ", " and "))
    println(io, " - dimensions: ", join(string.(COMPILED_DIMS) .* "D", ", ", " and "))
    println(io, " - layouts:    ", join(nameof.(COMPILED_LAYOUTS), ", ", " and "))

    if internal
        println(io, "\nKokkos internal configuration:")
        print_configuration(io, verbose)
    end
end


requirement_fail(msg, constraint::Base.Fix1, expected) = "$msg: $(constraint.x) $(nameof(constraint.f)) $expected"
requirement_fail(msg, constraint::Base.Fix2, expected) = "$msg: $expected $(nameof(constraint.f)) $(constraint.x)"
requirement_fail(msg, constraint::String, expected, value) = "$msg: $expected $constraint $value"
requirement_fail(msg, args...) = "$msg"


function require_config(version, dims, types, layouts, idx, exec_spaces, mem_spaces)
    failures = []

    if !isnothing(version)
        if KOKKOS_PATH != LOCAL_KOKKOS_DIR
            @warn "Cannot check the Kokkos version of a custom installation when Kokkos is not loaded" maxlog=1
        elseif !version(VersionNumber(LOCAL_KOKKOS_VERSION_STR))
            push!(failures, requirement_fail("Kokkos version $LOCAL_KOKKOS_VERSION_STR", version, "VERSION"))
        end
    end

    if !isnothing(idx)
        @warn "Cannot check the index type when Kokkos is not loaded" maxlog=1
    end

    if !isnothing(dims) && !issubset(dims, KOKKOS_VIEW_DIMS)
        push!(failures, requirement_fail("view dimensions", "⊈", tuple(dims...), KOKKOS_VIEW_DIMS))
    end

    if !isnothing(types)
        types_str = string.(nameof.(types))
        if !issubset(types_str, KOKKOS_VIEW_TYPES)
            push!(failures, requirement_fail("view types", "⊈", tuple(types_str...), KOKKOS_VIEW_TYPES))
        end
    end

    if !isnothing(layouts)
        layouts_str = layouts .|> nameof .|> string .|> lowercase
        layouts_str = chop.(layouts_str; head=length("Layout"), tail=0)
        layouts_str = string.(layouts_str)
        if !issubset(layouts_str, KOKKOS_VIEW_LAYOUTS)
            push!(failures, requirement_fail("view layouts", "⊈", tuple(layouts...), KOKKOS_VIEW_LAYOUTS))
        end
    end

    if !isnothing(exec_spaces)
        backends_str = string.(nameof.(exec_spaces))
        if !issubset(backends_str, KOKKOS_BACKENDS)
            push!(failures, requirement_fail("execution spaces", "⊈", tuple(backends_str...), KOKKOS_BACKENDS))
        end
    end

    if !isnothing(mem_spaces)
        @warn "Cannot check available memory spaces when Kokkos is not loaded" maxlog=1
    end

    return failures
end


function require_compiled(version, dims, types, layouts, idx, exec_spaces, mem_spaces)
    failures = []

    if !isnothing(version) && !version(KOKKOS_VERSION)
        push!(failures, requirement_fail("Kokkos version $KOKKOS_VERSION", version, "VERSION"))
    end

    if !isnothing(idx) && !idx(Idx)
        push!(failures, requirement_fail("index type $Idx", idx, "Idx"))
    end

    if !isnothing(dims) && !issubset(dims, COMPILED_DIMS)
        push!(failures, requirement_fail("view dimensions", "⊈", tuple(dims...), COMPILED_DIMS))
    end

    if !isnothing(types) && !issubset(types, COMPILED_TYPES)
        push!(failures, requirement_fail("view types", "⊈", tuple(types...), COMPILED_TYPES))
    end

    if !isnothing(layouts) && !issubset(layouts, COMPILED_LAYOUTS)
        push!(failures, requirement_fail("view layouts", "⊈", tuple(layouts...), COMPILED_LAYOUTS))
    end

    if !isnothing(exec_spaces) && !issubset(exec_spaces, COMPILED_EXEC_SPACES)
        push!(failures, requirement_fail("execution spaces", "⊈", tuple(exec_spaces...), COMPILED_EXEC_SPACES))
    end

    if !isnothing(mem_spaces) && !issubset(mem_spaces, COMPILED_MEM_SPACES)
        push!(failures, requirement_fail("memory spaces", "⊈", tuple(mem_spaces...), COMPILED_MEM_SPACES))
    end

    return failures
end


"""
    require(;
        version=nothing,
        dims=nothing, types=nothing, layouts=nothing,
        idx=nothing,
        exec_spaces=nothing, mem_spaces=nothing,
        no_error=false
    )

Asserts that the underlying Kokkos wrapper library of `Kokkos.jl` respects the given requirements.

An argument with a value of `nothing` is considered to have no requirements.

`version` checks for the version of Kokkos.

`idx` checks the type of the index variables used when accessing a view.

`version` and `idx` are given as callables returning a `Bool` and taking a single argument:
respectively a `VersionNumber` and a `Type`, e.g. passing `version = >=(v"4.0.0")` will match all
Kokkos versions including `v4.0.0` and above.

`dims`, `types`, `layouts`, `exec_spaces` and `mem_spaces` are lists of the required values. 

`dims`, `types` and `layouts` check the available dimensions, types and layouts of views, while
`exec_spaces` and `mem_spaces` do the same for execution and memory spaces.

If `no_error` is true, then this function will return `false` if any requirement is not met.

This function does not require for Kokkos to be initialized, but for the wrapper library to be
loaded.
If the wrapper is not loaded, the configuration options will be checked instead, however they cannot
cover all possible requirements (`idx` and `mem_spaces` do not work and `version` works only if the
packaged Kokkos installation is used).


# Examples

```julia
# Require Kokkos version 4.0.0 (exactly), and for 1D and 2D views of Float64 to be compiled with
# a column or row major layout:
Kokkos.require(;
    version = ==(v"4.0.0"),
    types = [Float64],
    dims = [1, 2],
    layouts = [Kokkos.LayoutLeft, Kokkos.LayoutRight]    
)

# Require an index type of 8 bytes, the Cuda and OpenMP backends of Kokkos, as well as the Cuda UVM
# memory space to be available:
Kokkos.require(;
    idx = (==(8) ∘ sizeof),
    exec_spaces = [Kokkos.Cuda, Kokkos.OpenMP],
    mem_spaces = [Kokkos.CudaUVMSpace]
)
```
"""
function require(;
    version=nothing,
    dims=nothing, types=nothing, layouts=nothing,
    idx=nothing,
    exec_spaces=nothing, mem_spaces=nothing,
    no_error=false
)
    if is_kokkos_wrapper_loaded()
        failures = require_compiled(version, dims, types, layouts, idx, exec_spaces, mem_spaces)
    else
        failures = require_config(version, dims, types, layouts, idx, exec_spaces, mem_spaces)
    end

    if !isempty(failures)
        no_error && return false
        config_str = is_kokkos_wrapper_loaded() ? "" : "configuration"
        length(failures) == 1 && error("$config_str requirement not met for $(first(failures))")
        error("$config_str requirements not met for:\n" * join(" - " .* failures, "\n"))
    end

    return true
end


"""
    KOKKOS_VERSION::VersionNumber

The Kokkos version currently loaded.

`nothing` if Kokkos is not yet loaded.
See [kokkos_version](@ref) for the version of the packaged installation of Kokkos, which
is defined before loading Kokkos.
"""
KOKKOS_VERSION = nothing


function __init_vars()
    impl = get_impl_module()
    global KOKKOS_VERSION = Base.invokelatest(impl.__kokkos_version)
end
