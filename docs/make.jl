
push!(LOAD_PATH, joinpath(@__DIR__, "../"))

using Kokkos
using Documenter
using CUDA  # For the extension documentation

ci = get(ENV, "CI", "") == "true"

DocMeta.setdocmeta!(Kokkos, :DocTestSetup, :(using Kokkos); recursive=true)

makedocs(;
    modules=[Kokkos],
    authors="Keluaa <34173752+Keluaa@users.noreply.github.com> and contributors",
    repo="https://github.com/Keluaa/Kokkos.jl/blob/{commit}{path}#{line}",
    sitename="Kokkos.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => [
            "Calling a Kokkos library" => "calling_c.md",
            "Using views in an inaccessible memory space" => "inaccessible_views.md",
            "Interoperability with CUDA.jl" => "interop.md",
            "MPI" => "MPI.md"
        ],
        "Environment" => "environment.md",
        "Execution & Memory Spaces" => "spaces.md",
        "Views" => "views.md",
        "Compilation" => "compilation.md",
        "Library Management" => "library_management.md",
        "Configuration options" => "config_options.md"
    ],
)


if ci
    deploydocs(
        repo = "github.com/Keluaa/Kokkos.jl.git",
        push_preview = true
    )
end
