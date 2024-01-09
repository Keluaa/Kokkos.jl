module DynamicCompilation

import ..Kokkos: Wrapper
import ..Kokkos: CMakeKokkosProject, CLibrary
import ..Kokkos: ensure_kokkos_wrapper_loaded, compile, lib_path, load_lib, __validate_parameters

import FileWatching.Pidfile: mkpidlock


export @compile_and_call, compile_and_load


const COMPILATION_LOCK_FILE = "__compilation.lock"
const PROCESS_ID = rand(Cint)  # getpid() is not guaranteed to be unique in an MPI app
const INTRA_PROCESS_LOCK = ReentrantLock()  # Multi-threading lock


const LOADED_FUNCTION_LIBS = Dict{String, CLibrary}()
const NEXT_LIB_NUMBER = Ref(0)
const LOAD_LIB_LOCK = ReentrantLock()


# CMake targets to their output
const CMAKE_TARGETS = Dict{String, String}(
    "views" => "libviews_out",
    "copy" => "libcopy_out",
    "mirrors" => "libmirrors_out",
    "subviews" => "libsubviews_out"
)


const SHARED_LIB_EXT = @static if Sys.iswindows()
    ".dll"
elseif Sys.isapple()
    ".dylib"
else
    ".so"
end


"""
    clean_libs()

Remove all shared libraries generated for compilation on demand of some methods or types.

Libraries are not unloaded, therefore subsequent calls to [`compile_and_load`](@ref) might not
trigger recompilation.
"""
function clean_libs()
    for file in readdir(Wrapper.get_kokkos_func_libs_dir(); join=true)
        !endswith(file, SHARED_LIB_EXT) && continue
        rm(file)
    end
end


const MPI_PKG_ID = Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI")
is_MPI_loaded() = !isnothing(get(Base.loaded_modules, MPI_PKG_ID, nothing))


function use_compilation_lock()
    user_opinion = get(ENV, "JULIA_KOKKOS_USE_COMPILATION_LOCK_FILE", nothing)
    !isnothing(user_opinion) && return user_opinion
    return is_MPI_loaded()
end


lock_file_path() = joinpath(Wrapper.KOKKOS_BUILD_DIR, COMPILATION_LOCK_FILE)

Wrapper.clear_compilation_lock() = rm(lock_file_path(); force=true)


"""
    compilation_lock(func)

Asserts that only a single process (and single thread) is compiling at once.

By default, there is only a lock on tasks of the current process.

If `MPI.jl` is loaded, a PID lock file is also used
(see [FileWatching.Pidfile](https://docs.julialang.org/en/v1/stdlib/FileWatching/#Pidfile),
or [Pidfile.jl](https://github.com/vtjnash/Pidfile.jl) before 1.9).
Lock files have the advantage that no collective MPI operations are needed, however they only work
if all MPI ranks share the same filesystem, and if this is not the case then there is no need for
lock files.
This can be overloaded with the `JULIA_KOKKOS_USE_COMPILATION_LOCK_FILE` environment variable.

Note that it is not the current process' ID that is used in the PID lock file, since PID are not
guaranteed to be unique in a MPI application. Instead a random 32-bit number is used, constant for
this process.
"""
function compilation_lock(func)
    lock(INTRA_PROCESS_LOCK) do
        !use_compilation_lock() && return func()
        return mkpidlock(func, lock_file_path(), PROCESS_ID; stale_age=0, wait=true, poll_interval=5)
    end
end


function build_lib_name(
    cmake_target,
    view_layout, view_dim, view_type,
    exec_space, mem_space,
    dest_layout, dest_space,
    without_exec_space_arg, with_nothing_arg,
    subview_dim
)
    # All arguments are strings
    parts = [cmake_target]

    !isempty(view_dim)     && push!(parts, view_dim .* "D")
    !isempty(view_layout)  && push!(parts, uppercase(view_layout[1:1]))  # Keep only the first letter
    !isempty(view_type)    && push!(parts, view_type)
    !isempty(exec_space)   && push!(parts, exec_space)
    !isempty(mem_space)    && push!(parts, mem_space)

    if !isempty(dest_layout) || !isempty(dest_space) || !isempty(subview_dim)
        push!(parts, "to")
    end

    !isempty(dest_layout)  && push!(parts, uppercase(dest_layout[1:1]))
    !isempty(dest_space)   && push!(parts, dest_space)
    !isempty(subview_dim)  && push!(parts, subview_dim)

    without_exec_space_arg && push!(parts, "no_exec")
    with_nothing_arg       && push!(parts, "with_default")

    return join(parts, "_") * SHARED_LIB_EXT
end


function is_lib_up_to_date(lib_path)
    !isfile(lib_path) && return false
    ensure_kokkos_wrapper_loaded()
    wrapper_lib_path = Wrapper.KOKKOS_LIB_PATH * SHARED_LIB_EXT
    !isfile(wrapper_lib_path) && error("Could not find the wrapper lib at '$wrapper_lib_path'")
    return mtime(wrapper_lib_path) ≤ mtime(lib_path)
end


function build_compilation_parameters(
    view_layout, view_dim, view_type,
    exec_space, mem_space,
    dest_layout, dest_space,
    without_exec_space_arg, with_nothing_arg,
    subview_dim
)
    # Special case for parameters which should have a default value
    dest_layout = isempty(dest_layout) ? "NONE" : dest_layout
    subview_dim = isempty(subview_dim) ? "0"    : subview_dim

    # Those are environment variables which will their respective macros in the C++ lib.
    # See 'lib/kokkos_wrapper/build_parameters.sh'
    return Dict(
        "VIEW_LAYOUT" => view_layout,
        "VIEW_DIM" => view_dim,
        "VIEW_TYPE" => view_type,
        "EXEC_SPACE" => exec_space,
        "MEM_SPACE" => mem_space,
        "DEST_LAYOUT" => dest_layout,
        "DEST_MEM_SPACE" => dest_space,
        "WITHOUT_EXEC_SPACE_ARG" => Int(without_exec_space_arg),
        "WITH_NOTHING_ARG" => Int(with_nothing_arg),
        "SUBVIEW_DIM" => subview_dim
    )
end


function compile_lib(cmake_target, out_lib_path, parameters)
    ensure_kokkos_wrapper_loaded()
    compilation_lock() do
        # The lock may have been acquired after another process compiled the library, therefore we
        # must check if the library is up to date again.
        is_lib_up_to_date(out_lib_path) && return

        target_output = CMAKE_TARGETS[cmake_target]

        lib_proj = CMakeKokkosProject(Wrapper.KOKKOS_LIB_PROJECT, cmake_target, target_output)
        compile(lib_proj; cmd_transform=cmd -> addenv(cmd, parameters))

        output_path = lib_path(lib_proj) * SHARED_LIB_EXT
        mv(output_path, out_lib_path; force=true)
    end
end


function register_new_functions(current_module, lib_path, lib_name)
    lock(LOAD_LIB_LOCK) do
        # Get the function pointer from the lib
        func_lib = load_lib(lib_path)
        LOADED_FUNCTION_LIBS[lib_name] = func_lib

        lib_number = (NEXT_LIB_NUMBER[] += 1)
        name = Symbol("Impl_", lib_number)

        module_expr = quote
            module $name
                using CxxWrap
                @static if pkgversion(CxxWrap) >= v"0.14-"
                    @wrapmodule(() -> $lib_path, :define_kokkos_module)
                else
                    @wrapmodule($lib_path, :define_kokkos_module)
                end
            end
        end
        module_expr = module_expr.args[2]  # Needed because of the error '"module" expression not at top level'

        Core.eval(current_module, module_expr)
    end
end


"""
    compile_and_load(current_module, cmake_target; kwargs...)

Check if the library of `cmake_target` compiled with `kwargs` exists, if not compile it, then load
it.

The library is a CxxWrap module, which is then loaded into `current_module` in the sub-module
`Impl<number>` with '<number>' the total count of calls to `compile_and_load` in this Julia session.
"""
function compile_and_load(current_module, cmake_target;
    view_layout = nothing,
    view_dim = nothing,
    view_type = nothing,
    exec_space = nothing,
    mem_space = nothing,
    dest_layout = nothing,
    dest_space = nothing,
    without_exec_space_arg = false,
    with_nothing_arg = false,
    subview_dim = nothing
)
    view_layout, view_dim, view_type,
        exec_space, mem_space,
        dest_layout, dest_space,
        subview_dim = __validate_parameters(;
            view_layout, view_dim, view_type,
            exec_space, mem_space,
            dest_layout, dest_space,
            subview_dim
    )

    @debug "Compiling $cmake_target with:\n\t$(join([
        "view_layout = $view_layout",
        "view_dim = $view_dim",
        "view_type = $view_type",
        "exec_space = $exec_space",
        "mem_space = $mem_space",
        "dest_layout = $dest_layout",
        "dest_space = $dest_space",
        "without_exec_space_arg = $without_exec_space_arg",
        "with_nothing_arg = $with_nothing_arg",
        "subview_dim = $subview_dim"
    ], "\n\t"))"

    # The lib name must uniquely identify a compilation with its parameters, in order to be able to
    # reuse a previously compiled lib safely.
    lib_name = build_lib_name(
        cmake_target,
        view_layout, view_dim, view_type,
        exec_space, mem_space,
        dest_layout, dest_space,
        without_exec_space_arg, with_nothing_arg,
        subview_dim
    )

    lib_path = joinpath(Wrapper.get_kokkos_func_libs_dir(), lib_name)

    if !is_lib_up_to_date(lib_path)
        parameters = build_compilation_parameters(
            view_layout, view_dim, view_type,
            exec_space, mem_space,
            dest_layout, dest_space,
            without_exec_space_arg, with_nothing_arg,
            subview_dim
        )
        @debug "Building '$cmake_target' in lib at '$lib_path'"
        compile_lib(cmake_target, lib_path, parameters)
    else
        @debug "Getting '$cmake_target' in lib from '$lib_path' (already compiled)"
    end

    register_new_functions(current_module, lib_path, lib_name)
end


"""
    call_more_specific(func, args)

Call `func` with `args` with `Base.invokelatest(func, args...)`, but only if there is a specialized
method for the arguments (an error is raised otherwise).

After calling [`compile_and_load`](@ref) from a `@nospecialize` method meant to define a new method
specialized for `args`, this function will prevent infinite recursion if the new method is not
applicable to `args`.
"""
function call_more_specific(@nospecialize(func), @nospecialize(args))
    return Base.invokelatest(call_more_specific_in_newest_world, func, args)
end


function latest_world_has_more_specific(@nospecialize(func), @nospecialize(args))
    return Base.invokelatest(has_specialization, func, Base.typesof(args...))
end


"""
    has_specialization(func, args_t::Tuple{Vararg{Type}})

True if the most specific method of `func` applicable to `args_t` has no `@nospecialize` annotation
on any argument.
"""
function has_specialization(@nospecialize(func), @nospecialize(args_t))
    try
        mth = which(func, args_t)
        return mth.nospecialize == 0
    catch
        return false
    end
end


function call_more_specific_in_newest_world(@nospecialize(func), @nospecialize(args))
    if !has_specialization(func, Base.typesof(args...))
        error("No specific method of $func for arguments $(Base.typesof(args...)): \
               $func failed to create an applicable method")
    end
    return func(args...)
end


"""
    @compile_and_call(method, args, compile)

If [`has_specialization(method, args)`](@ref has_specialization) then invoke `method`,
otherwise `compile` is executed.
In both cases, `method` is invoked through `Base.invokelatest`.

This handles the following case:
```
function my_method(@nospecialize(x))
    return @compile_and_call(my_method, x, begin #= ... =# end)
end

function my_program(x)
    my_method(x)  # Will compile
    my_method(x)
end
```

Here the second invocation of `my_method` is still done in the same world in which `my_program`
was invoked, even though the first invocation increased the global world counter.
Therefore, we need to ensure `method` has not been specialized in the latest world before
trying to compile.
"""
macro compile_and_call(method, args, compile)
    return esc(quote
        if $latest_world_has_more_specific($method, $args)
            Base.invokelatest($method, $args...)
        else
            $compile
            $call_more_specific($method, $args)
        end
    end)
end


end
