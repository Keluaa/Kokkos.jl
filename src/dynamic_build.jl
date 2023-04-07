
"""
    KokkosProject

Abstract type representing a C++ project using Kokkos, where it is located and how to compile it
using which options.
"""
abstract type KokkosProject end


"""
    build_dir(project::KokkosProject)

Return the build directory of `project`.
"""
function build_dir end


"""
    source_dir(project::KokkosProject)

Return the source directory of `project`.
"""
function source_dir end


"""
    lib_path(project::KokkosProject)

Return the path to the target library for `project`.
"""
function lib_path end


"""
    options(project::KokkosProject)

Return the set of options to the target library for `project`.
"""
function options end


"""
    option!(project::KokkosProject, name::String, val; prefix="Kokkos_")

Sets the given Kokkos option for the `project` to `val`.
This will result in the following compilation option: `\$(prefix)\$(name)=\$(val)`.
"""
function option! end


"""
    configuration_changed(project::KokkosProject)

Return `true` if the configuration of `project` changed, and needs to be recompiled.
"""
function configuration_changed end


"""
    configuration_changed!(project::KokkosProject, val::Bool = true)

Sets the configuration state of `project` to `val`.
"""
function configuration_changed! end


function read_stdout_file(file)
    flush(file)
    seekstart(file)
    return read(file, String)
end


function run_cmd_print_on_error(cmd::Cmd)
    mktemp() do _, file
        @debug "Running `$cmd`"
        try
            run(pipeline(cmd, stdout=file#= , stderr=file =#))
        catch
            println("Command failed. Standard output:\n", read_stdout_file(file))
            rethrow()
        end
        @debug "Command stdout:\n$(read_stdout_file(file))"
    end
end


function pretty_cmd_string(cmd::Cmd)
    cmd_str = ""
    if !isnothing(cmd.env) && !isempty(cmd.env)
        if length(cmd.env) > 3
            cmd_str *= "<vars...>"
        else
            cmd_str *= join(cmd.env, "; ")
        end
        cmd_str *= "; "
    end
    !isempty(cmd.dir) && (cmd_str *= "cd '$(cmd.dir)' && ")
    cmd_str *= string(`$(cmd.exec)`)[2:end-1]
    return cmd_str
end


function Base.print(io::IO, project::KokkosProject)
    print(io, "KokkosProject{src: '$(source_dir(project))', build: '$(build_dir(project))'}")
end


function Base.show(io::IO, project::KokkosProject)
    print(io, "Kokkos project from sources located at '$(source_dir(project))', ")
    println(io, "building in '$(build_dir(project))'.")
    println(io, "The target library is at '$(lib_path(project))'.")
    opts = options(project)
    if !isempty(opts)
        println(io, "Kokkos options:")
        foreach(options(project)) do (option, value)
            println(io, " - $option = $value")
        end
    else
        println(io, "Kokkos options: <none>")
    end
    println("Config command:  `", pretty_cmd_string(configure_command(project)), "`")
    println("Compile command: `", pretty_cmd_string(compile_command(project)), "`")
    println("Clean command:   `", pretty_cmd_string(clean_command(project)), "`")
end


"""
    configure(project::KokkosProject)

Configure the project with its current options.
"""
function configure(project::KokkosProject)
    @debug "Configuring project at '$(source_dir(project))'"
    mkpath(build_dir(project))
    run_cmd_print_on_error(configure_command(project))
    configuration_changed!(project, false)
    return nothing
end


"""
    compile(project::KokkosProject)

Builds all source files of the project.

If the project's configuration changed, it is reconfigured first.
"""
function compile(project::KokkosProject)
    if configuration_changed(project)
        configure(project)
    end
    @debug "Compiling project at '$(source_dir(project))' to '$(build_dir(project))'"
    run_cmd_print_on_error(compile_command(project))
    return nothing
end


"""
    clean(project::KokkosProject; reset=false)

Clean the project, forcing a recompilation of all source files.

If `reset == true`, **the entire `build_dir` is removed**, therefore ensuring that no cached CMake
variable can interfere with the build.
"""
function clean(project::KokkosProject; reset=false)
    if reset
        b_dir = build_dir(project)
        @debug "Removing all build files at '$b_dir'"
        rm(b_dir; force=true, recursive=true)
        mkpath(b_dir)
        configuration_changed!(project)
    else
        @debug "Cleaning build files at '$(build_dir(project))'"
        run_cmd_print_on_error(clean_command(project))
    end
    return nothing
end


function pretty_compile(p::KokkosProject)
    # TODO: add loading bars?
    @info "Configuring Kokkos project at $(source_dir(p))"
    configure(p)
    @info "Compiling Kokkos project at $(source_dir(p))"
    compile(p)
end


mutable struct CMakeKokkosProject <: KokkosProject
    source_dir::String
    build_dir::String
    lib_path::String
    
    commands_dir::String

    config_options::Vector{String}
    build_options::Vector{String}
    clean_options::Vector{String}

    cmake_options::Vector{String}
    kokkos_options::Dict{String, String}

    configuration_changed::Bool
end


"""
    CMakeKokkosProject(source_dir, target_lib_path;
        target = "all",
        build_type = "Release",
        build_dir = joinpath(source_dir, "cmake-build-\$(lowercase(build_type))"),
        cmake_options = [],
        kokkos_path = nothing, kokkos_options = Dict{String, String}()
    )

Construct a new Kokkos project in `source_dir` built to `build_dir` using CMake.
After compilation, the target library can be found at `joinpath(build_dir, target_lib_path)`.

The shared library extension of `target_lib_path` can be omitted, as it is added if needed by
`Libdl.dlopen`.

`target` is the CMake target needed to build the library.

`build_type` controls the `CMAKE_BUILD_TYPE` variable, and `cmake_options` contains all other
options passed to each CMake command.

`kokkos_path` sets the `Kokkos_ROOT` CMake variable, to be used by
[`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html).

`kokkos_path` should be left to `nothing` in most cases, as it will be replaced by the installation
directory of Kokkos correctly configured with the current options (see
[`get_kokkos_install_dir`](@ref)).

!!! warning

    The Kokkos version of the project must match the version used by the internal wrapper library.
    If this is not the case, the program may silently fail.
    Use the [kokkos_path](@ref) configuration variable to change the Kokkos version throughout
    `Kokkos.jl`.

`kokkos_options` contains the list of Kokkos variables needed to configure the project
([see the docs](https://kokkos.github.io/kokkos-core-wiki/keywords.html#device-backends)).

All commands are invoked from the current working directory.
"""
function CMakeKokkosProject(source_dir, target_lib_path;
    target = "all",
    build_type = "Release",
    build_dir = joinpath(source_dir, "cmake-build-$(lowercase(build_type))"),
    cmake_options = String[],
    kokkos_path = nothing, kokkos_options = Dict{String, String}()
)
    source_dir = normpath(source_dir)
    build_dir = normpath(build_dir)
    lib_path = normpath(build_dir, target_lib_path) |> abspath

    if source_dir == build_dir
        error("source directory and build directory are equal, calling \
               `clean(project; reset=true)`will remove the sources as well!")
    end

    config_options = String[
        "-S", source_dir,
        "-B", build_dir
    ]

    build_options = String[
        "--build", build_dir,
        "--target", target,
        "-j"
    ]

    clean_options = String[
        "--build", build_dir,
        "--target", "clean"
    ]

    push!(cmake_options, "-DCMAKE_BUILD_TYPE=$build_type", "-DBUILD_SHARED_LIBS=ON")

    if isnothing(kokkos_path)
        push!(cmake_options, "-DKokkos_ROOT=$(get_kokkos_install_dir())")
    elseif !isempty(kokkos_path)
        push!(cmake_options, "-DKokkos_ROOT=$kokkos_path")
    end

    return CMakeKokkosProject(
        source_dir, build_dir, lib_path,
        pwd(),
        config_options, build_options, clean_options,
        cmake_options, kokkos_options,
        true
    )
end


"""
    CMakeKokkosProject(project::CMakeKokkosProject, target, target_lib_path)

Construct a project from another, for a different target.

The source and build directories will stay the same, and options will be shared.
"""
function CMakeKokkosProject(project::CMakeKokkosProject, target, target_lib_path)
    lib_path = normpath(project.build_dir, target_lib_path) |> abspath

    build_options = String[
        "--build", project.build_dir,
        "--target", target,
        "-j"
    ]

    return CMakeKokkosProject(
        project.source_dir, project.build_dir, lib_path,
        project.commands_dir,
        project.config_options, build_options, project.clean_options,
        project.cmake_options, project.kokkos_options,
        project.configuration_changed
    )
end


build_dir(p::CMakeKokkosProject) = p.build_dir
source_dir(p::CMakeKokkosProject) = p.source_dir
lib_path(p::CMakeKokkosProject) = p.lib_path

options(p::CMakeKokkosProject) = p.kokkos_options
function option!(p::CMakeKokkosProject, name::String, val; prefix="Kokkos_")
    p.kokkos_options[prefix * name] = val
    configuration_changed!(p)
end

configuration_changed(p::CMakeKokkosProject) = p.configuration_changed
configuration_changed!(p::CMakeKokkosProject, val::Bool = true) = (p.configuration_changed = val)


function configure_command(p::CMakeKokkosProject)
    options_list = []
    for (k, v) in p.kokkos_options
        push!(options_list, "-D$k=$v")
    end
    return Cmd(`cmake $(p.config_options) $(p.cmake_options) $(options_list)`; dir=p.commands_dir)
end


function compile_command(p::CMakeKokkosProject)
    return Cmd(`cmake $(p.build_options)`; dir=p.commands_dir)
end


function clean_command(p::CMakeKokkosProject)
    return Cmd(`cmake $(p.clean_options)`; dir=p.commands_dir)
end
