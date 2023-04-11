
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
"""
function set_omp_vars(; places = "cores", bind = "close", num_threads = Base.Threads.nthreads())
    ENV["OMP_PLACES"] = places
    ENV["OMP_PROC_BIND"] = bind
    ENV["OMP_NUM_THREADS"] = num_threads
    if haskey(ENV, "KMP_AFFINITY")
        # Prevent Intel's variables from interfering with ours
        delete!(ENV, "KMP_AFFINITY")
    end
end


function warn_config_changed()
    if HAS_CONFIGURATION_CHANGED
        @warn "Kokkos configuration changed! Restart Julia for those changes to take effect!" maxlog=1
    end
end


"""
    initialize(;
        num_threads=nothing,
        device_id=nothing, map_device_id_by=nothing,
        disable_warnings=nothing, print_configuration=nothing,
        tune_internals=nothing,
        tools_libs=nothing, tools_args=nothing
    )

Initializes Kokkos by calling `Kokkos::initialize()`.

The keyword arguments build are passed to the `InitializationSettings` constructor which is then
passed to `Kokkos::initialize()`. A value of `nothing` keeps the default behaviour of Kokkos.
"""
function initialize(;
    num_threads=nothing,
    device_id=nothing, map_device_id_by=nothing,
    disable_warnings=nothing, print_configuration=nothing,
    tune_internals=nothing,
    tools_libs=nothing, tools_args=nothing
)
    warn_config_changed()
    settings = InitializationSettings()
    !isnothing(num_threads)         && num_threads!(settings, num_threads)
    !isnothing(device_id)           && device_id!(settings, device_id)
    !isnothing(disable_warnings)    && disable_warnings!(settings, disable_warnings)
    !isnothing(print_configuration) && print_configuration!(settings, print_configuration)
    !isnothing(tune_internals)      && tune_internals!(settings, tune_internals)
    !isnothing(tools_libs)          && tools_libs!(settings, tools_libs)
    !isnothing(tools_args)          && tools_args!(settings, tools_args)
    !isnothing(map_device_id_by)    && map_device_id_by!(settings, map_device_id_by)
    initialize(settings)
end


"""
    finalize()

Calls `Kokkos::finalize()`.

!!! warning

    Kokkos requires that all view destructors should be called __before__ calling `finalize`.
"""
function finalize end


"""
    is_initialized()

Return `Kokkos::is_initialized()`.
"""
function is_initialized end


"""
    is_finalized()

Return `Kokkos::is_finalized()`.
"""
function is_finalized end


"""
    versioninfo(io::IO = stdout; internal=true, verbose=false)

Print the version and various information about the underlying Kokkos library.

If `internal` is true, `Kokkos::print_configuration()` is called. `verbose` is passed to that call.

This function does not require for Kokkos to be initialized, however if `internal=true` then the
output will be incomplete.
"""
function versioninfo(io::IO = stdout; internal=true, verbose=false)
    warn_config_changed()
    print(io, "Kokkos version $KOKKOS_VERSION")
    if isnothing(KOKKOS_PATH)
        println(io, " (packaged sources)")
    else
        println(io, " (path: $(KOKKOS_PATH))")
    end
    println(io, "Kokkos installation dir: ", get_kokkos_install_dir())
    println(io, "Wrapper library compiled at ", build_dir(KokkosWrapper.KOKKOS_LIB_PROJECT))
    println(io, "Compiled execution spaces:")
    for space in COMPILED_EXEC_SPACES
        println(io, " - $(nameof(space)) (default memory space: $(nameof(memory_space(space))))")
    end
    println(io, "Compiled memory spaces:")
    for space in COMPILED_MEM_SPACES
        println(io, " - $(nameof(space)) (associated execution space: $(nameof(execution_space(space))))")
    end
    println(io, "Compiled view types: ", join(COMPILED_TYPES, ", ", " and "))
    println(io, "Compiled view dimensions: ", join(string.(COMPILED_DIMS) .* "D", ", ", " and "))

    if internal
        println(io, "\nKokkos internal configuration:")
        print_configuration(io, verbose)
    end
end


requirement_fail(msg, constraint::Base.Fix1, expected) = "$msg: $(constraint.x) $(nameof(constraint.f)) $expected"
requirement_fail(msg, constraint::Base.Fix2, expected) = "$msg: $expected $(nameof(constraint.f)) $(constraint.x)"
requirement_fail(msg, constraint::String, expected, value) = "$msg: $expected $constraint $value"
requirement_fail(msg, args...) = "$msg"


"""
    require(;
        version=nothing,
        dims=nothing, types=nothing, idx=nothing,
        exec_spaces=nothing, mem_spaces=nothing,
        no_error=false
    )

Asserts that the underlying Kokkos library of `Kokkos.jl` respects the given requirements.

An argument with a value of `nothing` is considered to have no requirements.

`version` checks for the version of Kokkos.

`idx` checks the type of the index variables used when accessing a view.

`version` and `idx` are given as callables returning a `Bool` and taking a single argument:
respectively a `VersionNumber` and a `Type`, e.g. passing `version = >=(v"4.0.0")` will match all
Kokkos versions including `v4.0.0` and above.

`dims`, `types`, `exec_spaces` and `mem_spaces` are lists of the required values. 

`dims` and `types` check the available dimensions and types of views, while `exec_spaces` and
`mem_spaces` do the same for execution and memory spaces.

If `no_error` is true, then this function will return `false` if any requirement is not met.

This function does not require for Kokkos to be initialized.

# Examples

```julia
# Require Kokkos version 4.0.0 (exactly), and for 1D and 2D views of Float64 to be compiled:  
Kokkos.require(;
    version = ==(v"4.0.0"),
    types = [Float64],
    dims = [1, 2]
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
    dims=nothing, types=nothing, idx=nothing,
    exec_spaces=nothing, mem_spaces=nothing,
    no_error=false
)
    failures = []

    if !isnothing(version) && !version(KOKKOS_VERSION)
        push!(failures, requirement_fail("Kokkos version $KOKKOS_VERSION", version, "VERSION"))
    end

    if !isnothing(idx) && !idx(Idx)
        push!(failures, requirement_fail("index type $Idx", idx, "Idx"))
    end

    if !isnothing(dims) && !issubset(dims, COMPILED_DIMS)
        push!(failures, requirement_fail("view dimensions", "⊆", tuple(dims...), COMPILED_DIMS))
    end

    if !isnothing(types) && !issubset(types, COMPILED_TYPES)
        push!(failures, requirement_fail("view types", "⊆", tuple(types...), COMPILED_TYPES))
    end

    if !isnothing(exec_spaces) && !issubset(exec_spaces, COMPILED_EXEC_SPACES)
        push!(failures, requirement_fail("execution spaces", "⊆", tuple(exec_spaces...), COMPILED_EXEC_SPACES))
    end

    if !isnothing(mem_spaces) && !issubset(mem_spaces, COMPILED_MEM_SPACES)
        push!(failures, requirement_fail("memory spaces", "⊆", tuple(mem_spaces...), COMPILED_MEM_SPACES))
    end

    if !isempty(failures)
        no_error && return false
        length(failures) == 1 && error("requirement not met for $(first(failures))")
        error("requirements not met for:\n" * join(" - " .* failures, "\n"))
    end

    return true
end
