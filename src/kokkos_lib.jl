
mutable struct CLibrary
    load_path::String
    full_path::String
    handle::Ptr{Cvoid}
    symbols::Dict{Symbol, Ptr{Cvoid}}
end


is_valid(lib::CLibrary) = lib.handle != C_NULL


const PROJECTS_LIBS = Dict{String, CLibrary}()


"""
    load_lib(lib::Union{String, KokkosProject, CLibrary};
             flags=Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_LOCAL)

Open a shared library.

If the argument is a project, its target is supposed to be compiled and up-to-date.

If the library is already loaded, it is not opened another time: this guarantees that calling
`Libdl.dlclose` will unload the library from memory, if the library wasn't opened from elsewhere.
"""
function load_lib(lib::CLibrary; flags=Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_LOCAL)
    if !is_valid(lib)
        lib.handle = Libdl.dlopen(lib.load_path, flags)
        lib.full_path = Libdl.dlpath(lib.handle)
    end
    return lib
end

function load_lib(path::String; kwargs...)
    lib = get!(PROJECTS_LIBS, path) do; CLibrary(path, "", C_NULL, Dict()) end
    return load_lib(lib; kwargs...)
end

load_lib(project::KokkosProject; kwargs...) = load_lib(lib_path(project); kwargs...)


"""
    unload_lib(lib::Union{String, KokkosProject, CLibrary})

Unload a library. Return true if the library has a valid handle and `Libdl.dlclose` was called.

Because of the mechanism behind shared library loading, it is not guaranteed that the library is
unloaded from memory after this call. Using `is_lib_loaded` rather than the return value of this
function is more reliable.

The symbol cache of the library is cleared by this function.
"""
function unload_lib(lib::CLibrary)
    !is_valid(lib) && return false
    Libdl.dlclose(lib.handle)
    lib.handle = C_NULL
    empty!(lib.symbols)
    return true
end

function unload_lib(path::String)
    !haskey(PROJECTS_LIBS, path) && return false
    return unload_lib(PROJECTS_LIBS[path])
end

unload_lib(project::KokkosProject) = unload_lib(lib_path(project))


"""
    is_lib_loaded(lib::Union{String, KokkosProject, CLibrary})

Return `true` if the library was previously loaded by `load_lib` and is still present in memory.
"""
function is_lib_loaded(lib::CLibrary)
    is_valid(lib) && return true
    return findfirst(==(lib.full_path), Libdl.dllist()) |> !isnothing
end

function is_lib_loaded(path::String)
    !haskey(PROJECTS_LIBS, path) && return false
    return is_lib_loaded(PROJECTS_LIBS[path])
end

is_lib_loaded(project::KokkosProject) = is_lib_loaded(lib_path(project))


"""
    get_symbol(lib::CLibrary, symbol::Symbol)

Load the pointer to the given symbol. Symbol pointers are cached: `Libdl.dlsym` is called only if
the symbol is not already in the cache.
"""
function get_symbol(lib::CLibrary, symbol::Symbol)
    return get!(lib.symbols, symbol) do
        !is_valid(lib) && error("the library is not loaded")
        Libdl.dlsym(lib.handle, symbol)
    end
end
