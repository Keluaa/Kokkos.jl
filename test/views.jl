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
@test v1 isa Kokkos.KokkosWrapper.Impl.View1D_HostAllocated{Float64}
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

v6 = View{Float64, 2}(undef, (1, 2))
v6_simili = [
    View{Float64, 2, Kokkos.HostSpace}(undef, (1, 2)),
    View{Float64, 2, Kokkos.HostSpace}(undef, 1, 2),
    View{Float64, 2, Kokkos.HostSpace}((1, 2)),
    View{Float64, 2, Kokkos.HostSpace}(1, 2),
    View{Float64, 2}(undef, (1, 2)),
    View{Float64, 2}(undef, 1, 2),
    View{Float64, 2}((1, 2)),
    View{Float64, 2}(1, 2),
    View{Float64}(undef, (1, 2)),
    View{Float64}(undef, 1, 2),
    View{Float64}((1, 2)),
    View{Float64}(1, 2),
    View{Float64}(1, 2; dim_pad=true),
    View{Float64}(1, 2; label=""),
    View{Float64}(1, 2; mem_space=Kokkos.HostSpace)
]
for v6_s in v6_simili
    @test typeof(v6) == typeof(v6_s)
    @test size(v6) == size(v6_s)
end

@test size(View{Float64}()) == (0,)
@test size(View{Float64, 2}()) == (0, 0)

@test_throws @error_match("`Int32` is not compiled") View{Int32}(undef, n1)
@test_throws @error_match("`Kokkos.View3D` cannot") View{Int64}(undef, (2, 2, 2))
@test_throws @error_match("CudaSpace` is not compiled") View{Int64}(undef, n1; mem_space=Kokkos.CudaSpace)


@testset "view_wrap" begin
    a7 = rand(Float64, 43)
    v7 = view_wrap(a7)
    @test all(v7 .== a7)
    @test pointer(a7) == Kokkos.view_data(v7)
    
    a8 = rand(Float64, 16)
    v8 = view_wrap(View{Float64, 2, Kokkos.DEFAULT_HOST_MEM_SPACE}, (4, 4), pointer(a8))
    @test v8[4, 4] == a8[16]
    @test all(reshape(v8', 16) .== a8)  # HostSpace defaults to row-major, hence the transposition
    @test pointer(a8) == Kokkos.view_data(v8) 
end


@testset "Deep copy on $exec_space_type in $(dim)D" for 
        exec_space_type in (:no_exec_space, Kokkos.COMPILED_EXEC_SPACES...),
        dim in Kokkos.COMPILED_DIMS

    exec_space = exec_space_type === :no_exec_space ? nothing : Kokkos.impl_space_type(exec_space_type)()
    n = ntuple(Returns(7), dim)

    @testset "View{$src_type, $dim, $src_space} => View{$dst_type, $dim, $dst_space}" for 
            src_type in Kokkos.COMPILED_TYPES, dst_type in Kokkos.COMPILED_TYPES,
            src_space in Kokkos.COMPILED_MEM_SPACES, dst_space in Kokkos.COMPILED_MEM_SPACES

        v_src = View{src_type}(undef, n; mem_space=src_space)
        v_dst = View{dst_type}(n; mem_space=dst_space)

        v_src .= rand(src_type, n) * 10

        if isnothing(exec_space)
            Kokkos.deep_copy(v_dst, v_src)
        else
            Kokkos.deep_copy(exec_space, v_dst, v_src)
            Kokkos.fence(exec_space)
        end
    
        if src_type == dst_type
            @test all(v_dst .== v_src)
        elseif dst_type <: AbstractFloat
            @test all(v_dst .≈ v_src)  # Int to Float
        else
            @test all(v_dst .== trunc.(dst_type, v_src))  # Float to Int
        end
    end
end


@testset "create_mirror to $dst_space_type in $(dim)D" for 
        dst_space_type in (:default_mem_space, Kokkos.COMPILED_MEM_SPACES...),
        dim in Kokkos.COMPILED_DIMS

    dst_mem_space = dst_space_type === :default_mem_space ? nothing : Kokkos.impl_space_type(dst_space_type)()
    n = ntuple(Returns(7), dim)

    @testset "View{$src_type, $dim, $src_space}" for
            src_type in Kokkos.COMPILED_TYPES,
            src_space in Kokkos.COMPILED_MEM_SPACES
        v_src = View{src_type}(undef, n; mem_space=src_space)
        v_src_m = Kokkos.create_mirror(v_src; mem_space=dst_mem_space, zero_fill=true)

        @test all(v_src_m .== 0)

        if isnothing(dst_mem_space)
            @test Kokkos.accessible(Kokkos.memory_space(v_src_m))
        else
            @test Kokkos.memory_space(v_src_m) == Kokkos.main_space_type(dst_mem_space)
        end

        v_src_m2 = Kokkos.create_mirror_view(v_src; mem_space=dst_mem_space, zero_fill=true)

        if isnothing(dst_mem_space)
            @test Kokkos.accessible(Kokkos.memory_space(v_src_m2))
        elseif Kokkos.memory_space(v_src) == Kokkos.main_space_type(dst_mem_space)
            @test v_src_m2 == v_src
            @test pointer(v_src_m2) == pointer(v_src)
        else
            @test v_src_m2 != v_src
            @test pointer(v_src_m2) != pointer(v_src)
            @test all(v_src_m2 .== 0)
        end
    end
end

end