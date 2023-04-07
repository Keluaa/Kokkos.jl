@testset "Views" begin

import Kokkos: Idx, View

@test Idx <: Union{UInt64, Int64, UInt32, Int32}

@test Kokkos.COMPILED_DIMS == (1, 2)
@test Kokkos.COMPILED_TYPES == (Float64, Int64)

@test View <: AbstractArray
@test View{Float64} <: AbstractArray{Float64}
@test View{Float64, 2} <: AbstractArray{Float64, 2}
@test View{Float64, 2, Kokkos.HostSpace} <: AbstractArray{Float64, 2}
@test View{Float64, 3, Kokkos.HostSpace} <: AbstractArray{Float64, 3}

n1 = 10
v1 = View{Float64}(n1)

@test v1 isa View
@test v1 isa View{Float64}
@test v1 isa View{Float64, 1}
@test v1 isa View{Float64, 1, <:Kokkos.HostSpace}
@test v1 isa Kokkos.Views.View1D_HostAllocated{Float64}
@test v1 isa AbstractArray
@test v1 isa AbstractArray{Float64}
@test v1 isa AbstractArray{Float64, 1}

@test accessible(v1)
@test label(v1) == ""

@test size(v1) == (n1,)
@test length(v1) == n1
@test ndims(v1) == 1

v1_ptr = Kokkos.Views.get_ptr(v1, 0).cpp_object
@test Kokkos.Views.elem_ptr(v1, 1) == v1_ptr
@test Kokkos.Views.elem_ptr(v1, 2) == (v1_ptr + sizeof(Float64))
@test v1[1] == unsafe_load(v1_ptr, 1)
@test v1[2] == unsafe_load(v1_ptr + sizeof(Float64), 1)

@test_throws BoundsError v1[n1+1]
@test_throws BoundsError v1[0]

@test all(v1 .== 0)
v1 .= 1:n1
@test all(v1 .== eachindex(v1))

v2 = v1
v2[1] = 5
@test v1[1] == v2[1]

n2 = n1 + 10
v2 = View{Float64}(undef, n2; label="v2")
@test label(v2) == "v2"

v2[1:n1] .= v1
@test all(v1 .== v2[1:n1])

n3 = (n1, n1)
v3 = View{Int64}(n3)

@test size(v3) == n3
@test length(v3) == prod(n3)
@test ndims(v3) == 2

v3_ptr = Kokkos.Views.get_ptr(v3, 0, 0).cpp_object
@test Kokkos.Views.elem_ptr(v3, 1, 1) == v3_ptr
@test Kokkos.Views.elem_ptr(v3, 1, 2) == (v3_ptr + sizeof(Int64))
@test Kokkos.Views.elem_ptr(v3, 2, 1) == (v3_ptr + sizeof(Int64) * n3[1])
@test Kokkos.Views.elem_ptr(v3, 2, 2) == (v3_ptr + sizeof(Int64) * (n3[1] + 1))
@test v3[1, 1] == unsafe_load(v3_ptr, 1)
@test v3[1, 2] == unsafe_load(v3_ptr + sizeof(Int64), 1)

@test all(v3 .== 0)

a3 = reshape(collect(1:prod(n3)), n3)'
v3 .= LinearIndices(a3)'

@test all(v3 .== a3)

@test_throws BoundsError v3[0, 0]
@test_throws BoundsError v3[0, 1]
@test_throws BoundsError v3[(n3 .+ (1, 0))...]
@test_throws BoundsError v3[(n3 .+ (0, 1))...]

v4 = similar(v3)
@test size(v4) == size(v3)
@test typeof(v4) == typeof(v3)

v5 = copy(v3')
@test typeof(v5) == typeof(v3)
@test all(v5 .== v3')

# TODO
# v6 = View{Float64, 2}(undef, (1, 2))
# v6 = View{Float64, 2}(undef, 1, 2)
# v6 = View{Float64, 2}((1, 2))
# v6 = View{Float64, 2}(1, 2)

@test_throws "`Int32` is not compiled" View{Int32}(undef, n1)
@test_throws "`Kokkos.View3D` cannot" View{Int64}(undef, (2, 2, 2))
@test_throws "CudaSpace` is not compiled" View{Int64}(undef, n1; mem_space=Kokkos.CudaSpace)

end