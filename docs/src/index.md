```@meta
CurrentModule = Kokkos
```

# Kokkos.jl

Documentation for [Kokkos](https://github.com/Keluaa/Kokkos.jl).


`Kokkos.jl` allows you to create `Kokkos::View` instances from Julia, to configure and compile a
Kokkos project or load an existing library, and call its functions and the Kokkos kernels it defines.

`Kokkos.jl` supports all backends of Kokkos.

[`View`](@ref) inherits the `AbstractArray` interface of Julia, and can therefore be used as a
normal `Array`.
All view accesses are done through calls to `Kokkos::View::operator()`, and therefore can access CPU
or GPU memory seamlessly.


!!! note

    Currently `Kokkos.jl` does not create Kokkos kernels (e.g. using `Kokkos::parallel_for`) since
    it would require automatic Julia to C++ code convertion.
    You also cannot run Julia code in Kokkos kernels, since Julia cannot be used in threads it
    doesn't own (or adopted, in Julia 1.9).
