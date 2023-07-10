
# Interoperability

## CUDA.jl

Views can be converted into `CuArray`s using `Base.unsafe_wrap`:

```@docs
Base.unsafe_wrap(::Type{CUDA.CuArray}, ::Kokkos.View)
```

And `CuArray`s can be converted into views with [`Kokkos.view_wrap`](@ref):

```julia-repl
julia> A = CuArray{Int64}(undef, 4, 4);

julia> CUDA.@allowscalar for i in eachindex(A)
           A[i] = i
       end

julia> A
4×4 CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}:
 1  5   9  13
 2  6  10  14
 3  7  11  15
 4  8  12  16

julia> A_v = Kokkos.view_wrap(A)
4×4 Kokkos.Views.View{Int64, 2, Kokkos.LayoutLeft, Kokkos.CudaSpace}: <inaccessible view>
```

`SubArray`s of `CuArray`s (or, more precisely, any `CUDA.StridedSubCuArray`), can also be converted
into views with a `LayoutStride`:

```julia-repl
julia> sub_A = @view A[2:3, 2:3];

julia> sub_A isa CUDA.StridedSubCuArray
true

julia> size(sub_A), strides(sub_A)
((2, 2), (1, 4))

julia> sub_A_v = Kokkos.view_wrap(sub_A)
2×2 Kokkos.Views.View{Int64, 2, Kokkos.LayoutStride, Kokkos.CudaSpace}: <inaccessible view>

julia> size(sub_A_v), strides(sub_A_v)
((2, 2), (1, 4))
```

Unlike `Kokkos.View`, it is possible to perform arithmetic operations on a `CuArray` from the host,
as well as indexing device memory (if permitted by `CUDA.allowscalar(true)` or `CUDA.@allowscalar`).

## AMDGPU.jl

Views can be converted into `ROCArray`s using `Base.unsafe_wrap`:

```@docs
Base.unsafe_wrap(::Type{AMDGPU.ROCArray}, ::Kokkos.View)
```

And `ROCArray`s can be converted into views with [`Kokkos.view_wrap`](@ref), in the same manner as
for `CuArray`s.
