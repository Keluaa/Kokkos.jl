```@meta
CurrentModule = Kokkos
```

# Kokkos.jl

Documentation for [Kokkos](https://github.com/Keluaa/Kokkos.jl).


`Kokkos.jl` allows you to create `Kokkos::View` instances from Julia, to configure and compile a Kokkos project or load an existing library, and call its functions.

`Kokkos.jl` supports all backends of Kokkos.

[`View`](@ref) inherit the `AbstractArray` interface in Julia, and can therefore be used as a normal `Array`.
All view accesses are done through calls to `Kokkos::View::operator()`, and therefore can access CPU or GPU memory seamlessly.


!!! note

    Currently `Kokkos.jl` cannot create Kokkos kernels (using `Kokkos::parallel_for`) since it must require C++ code generation.
    You also cannot run Julia code in Kokkos kernels, since Julia cannot be used in threads it doesn't own (or did not adopt, in v1.9).
