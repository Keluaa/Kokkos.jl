using Kokkos
using Documenter

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
    ],
)
