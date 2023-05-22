@testset "Views" begin

import Kokkos: Idx, View

@test Idx <: Union{UInt64, Int64, UInt32, Int32}

@test Kokkos.COMPILED_DIMS == (1, 2)
@test Kokkos.COMPILED_TYPES == (Float64, Int64)
@test Kokkos.COMPILED_LAYOUTS == (Kokkos.LayoutLeft, Kokkos.LayoutRight, Kokkos.LayoutStride)

@test View <: AbstractArray
@test View{Float64} <: AbstractArray{Float64}
@test View{Float64, 2} <: AbstractArray{Float64, 2}
@test View{Float64, 2, Kokkos.LayoutLeft} <: AbstractArray{Float64, 2}
@test View{Float64, 3, Kokkos.LayoutLeft, Kokkos.HostSpace} <: AbstractArray{Float64, 3}

n1 = 10
v1 = View{Float64}(n1)

@test v1 isa View
@test v1 isa View{Float64}
@test v1 isa View{Float64, 1}
@test v1 isa View{Float64, 1, <:Kokkos.Layout}
@test v1 isa View{Float64, 1, Kokkos.LayoutRight}
@test v1 isa View{Float64, 1, Kokkos.LayoutRight, <:Kokkos.HostSpace}
@test v1 isa Kokkos.KokkosWrapper.Impl.View1D_R_HostAllocated{Float64}
@test v1 isa AbstractArray
@test v1 isa AbstractArray{Float64}
@test v1 isa AbstractArray{Float64, 1}

@test accessible(v1)
@test label(v1) == ""

@test size(v1) == (n1,)
@test length(v1) == n1
@test ndims(v1) == 1
@test strides(v1) == (1,)
@test Base.elsize(v1) == sizeof(Float64)

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
v3 = View{Int64, 2, Kokkos.LayoutRight}(n3)

@test size(v3) == n3
@test length(v3) == prod(n3)
@test ndims(v3) == 2
@test strides(v3) == (n1, 1)
@test Base.elsize(v3) == sizeof(Float64)
@test sizeof(v3) == sizeof(Float64) * prod(n3)

v3_ptr = Kokkos.Views.get_ptr(v3, 0, 0).cpp_object
@test Kokkos.Views.elem_ptr(v3, 1, 1) == v3_ptr
@test Kokkos.Views.elem_ptr(v3, 1, 2) == (v3_ptr + sizeof(Int64))
@test Kokkos.Views.elem_ptr(v3, 2, 1) == (v3_ptr + sizeof(Int64) * n3[1])
@test Kokkos.Views.elem_ptr(v3, 2, 2) == (v3_ptr + sizeof(Int64) * (n3[1] + 1))

@test all(v3 .== 0)

a3 = reshape(collect(1:prod(n3)), n3)'
v3 .= LinearIndices(a3)'

@test v3[1, 1] == unsafe_load(v3_ptr, 1)
@test v3[1, 2] == unsafe_load(v3_ptr + sizeof(Int64), 1)
@test all(v3 .== a3)

@test_throws BoundsError v3[0, 0]
@test_throws BoundsError v3[0, 1]
@test_throws BoundsError v3[(n3 .+ (1, 0))...]
@test_throws BoundsError v3[(n3 .+ (0, 1))...]

@testset "LayoutLeft" begin
    v3_l = View{Int64, 2, Kokkos.LayoutLeft}(n3)

    @test size(v3_l) == n3
    @test length(v3_l) == prod(n3)
    @test ndims(v3_l) == 2
    @test strides(v3_l) == (1, n1)
    @test Base.elsize(v3_l) == sizeof(Float64)
    @test sizeof(v3_l) == sizeof(Float64) * prod(n3)

    v3_l_ptr = Kokkos.Views.get_ptr(v3_l, 0, 0).cpp_object
    @test Kokkos.Views.elem_ptr(v3_l, 1, 1) == v3_l_ptr
    @test Kokkos.Views.elem_ptr(v3_l, 1, 2) == (v3_l_ptr + sizeof(Int64) * n3[1])
    @test Kokkos.Views.elem_ptr(v3_l, 2, 1) == (v3_l_ptr + sizeof(Int64))
    @test Kokkos.Views.elem_ptr(v3_l, 2, 2) == (v3_l_ptr + sizeof(Int64) * (n3[1] + 1))
end

@testset "LayoutStride" begin
    s = (2, 2*n3[1] + 3)
    v3_s = View{Int64, 2, Kokkos.LayoutStride}(n3; layout=Kokkos.LayoutStride(s))

    @test size(v3_s) == n3
    @test length(v3_s) == prod(n3)
    @test ndims(v3_s) == 2
    @test strides(v3_s) == s
    @test Base.elsize(v3_s) == sizeof(Float64)
    @test sizeof(v3_s) == sizeof(Float64) * s[2] * n3[2]

    v3_s_ptr = Kokkos.Views.get_ptr(v3_s, 0, 0).cpp_object
    @test Kokkos.Views.elem_ptr(v3_s, 1, 1) == v3_s_ptr
    @test Kokkos.Views.elem_ptr(v3_s, 2, 1) == (v3_s_ptr + sizeof(Int64) * s[1])
    @test Kokkos.Views.elem_ptr(v3_s, 1, 2) == (v3_s_ptr + sizeof(Int64) * s[2])
    @test Kokkos.Views.elem_ptr(v3_s, 2, 2) == (v3_s_ptr + sizeof(Int64) * (s[1] + s[2]))        
end

v4 = similar(v3)
@test size(v4) == size(v3)
@test typeof(v4) == typeof(v3)
@test strides(v4) == strides(v3)

v5 = copy(v3')
@test typeof(v5) == typeof(v3)
@test all(v5 .== v3')

@test memory_space(v5) === Kokkos.HostSpace

# View constructors
v6 = View{Float64, 2}(undef, (1, 2))
v6_simili = [
    View{Float64, 2, Kokkos.LayoutRight, Kokkos.HostSpace}(undef, (1, 2)),
    View{Float64, 2, Kokkos.LayoutRight, Kokkos.HostSpace}(undef, 1, 2),
    View{Float64, 2, Kokkos.LayoutRight, Kokkos.HostSpace}((1, 2)),
    View{Float64, 2, Kokkos.LayoutRight, Kokkos.HostSpace}(1, 2),
    View{Float64, 2, Kokkos.LayoutRight}(undef, (1, 2)),
    View{Float64, 2, Kokkos.LayoutRight}(undef, 1, 2),
    View{Float64, 2, Kokkos.LayoutRight}((1, 2)),
    View{Float64, 2, Kokkos.LayoutRight}(1, 2),
    View{Float64, 2}(undef, (1, 2); layout=Kokkos.LayoutRight),
    View{Float64, 2}(undef, 1, 2; layout=Kokkos.LayoutRight),
    View{Float64, 2}((1, 2); layout=Kokkos.LayoutRight),
    View{Float64, 2}(1, 2; layout=Kokkos.LayoutRight),
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
    View{Float64}(1, 2; mem_space=Kokkos.HostSpace),
    View{Float64}(1, 2; mem_space=Kokkos.HostSpace()),
    View{Float64}(1, 2; mem_space=Kokkos.HostSpace, layout=Kokkos.LayoutRight),
    View{Float64}(1, 2; mem_space=Kokkos.HostSpace, layout=Kokkos.LayoutRight())
]
for v6_s in v6_simili
    @test typeof(v6) == typeof(v6_s)
    @test size(v6) == size(v6_s)
    @test strides(v6) == strides(v6_s)
end

@test size(View{Float64}()) == (0,)
@test size(View{Float64, 2}()) == (0, 0)
@test size(View{Float64, 2, Kokkos.LayoutRight}()) == (0, 0)
@test size(View{Float64, 2, Kokkos.LayoutRight, Kokkos.HostSpace}()) == (0, 0)

@test_throws @error_match("`Int32` is not compiled") View{Int32}(undef, n1)
@test_throws @error_match("`Kokkos.View3D` cannot") View{Int64}(undef, (2, 2, 2))
@test_throws @error_match("CudaSpace is not compiled") View{Int64}(undef, n1; mem_space=Kokkos.CudaSpace)
@test_throws @error_match("`mem_space` kwarg") View{Float64, 1, Kokkos.LayoutLeft, Kokkos.KokkosWrapper.Impl.HostSpaceImplDereferenced}(undef, n1; mem_space=Kokkos.HostSpace)
@test_throws @error_match("Kokkos.Views.LayoutLeft type") View{Float64, 1, Kokkos.LayoutLeft}(undef, n1; layout=Kokkos.LayoutRight)
@test_throws @error_match("requires a instance") View{Float64}(undef, n1; layout=Kokkos.LayoutStride)


@testset "view_wrap" begin
    a7 = rand(Float64, 43)
    v7 = view_wrap(a7)
    @test all(v7 .== a7)
    @test pointer(a7) == Kokkos.view_data(v7)
    @test Base.unsafe_convert(Ptr{Float64}, v7) == pointer(a7)

    a8 = rand(Float64, 16)
    v8 = view_wrap(View{Float64, 2, Kokkos.LayoutRight, Kokkos.DEFAULT_HOST_MEM_SPACE}, (4, 4), pointer(a8))
    @test v8[4, 4] == a8[16]
    @test all(reshape(v8', 16) .== a8)  # HostSpace defaults to row-major, hence the transposition
    @test pointer(a8) == Kokkos.view_data(v8)
    @test Base.unsafe_convert(Ptr{Float64}, v8) == pointer(a8)

    v8_corners = view_wrap(
        View{Float64, 2, Kokkos.LayoutStride, Kokkos.DEFAULT_HOST_MEM_SPACE},
        (2, 2), pointer(a8);
        layout=Kokkos.LayoutStride(3, 12)  # Column-major layout
    )
    @test v8_corners == reshape(a8, (4, 4))[1:3:end, 1:3:end]  # No transposition needed
    @test pointer(a8) == Kokkos.view_data(v8_corners)
end


@testset "Deep copy on $exec_space_type in $(dim)D with $type" for 
        exec_space_type in (:no_exec_space, Kokkos.COMPILED_EXEC_SPACES...),
        dim in Kokkos.COMPILED_DIMS,
        type in Kokkos.COMPILED_TYPES

    exec_space = exec_space_type === :no_exec_space ? nothing : exec_space_type()
    n = ntuple(Returns(7), dim)

    @testset "View{$type, $dim, $src_layout, $src_space} => View{$type, $dim, $dst_layout, $dst_space}" for
            src_space in Kokkos.COMPILED_MEM_SPACES, dst_space in Kokkos.COMPILED_MEM_SPACES,
            src_layout in Kokkos.COMPILED_LAYOUTS, dst_layout in Kokkos.COMPILED_LAYOUTS

        if src_layout == Kokkos.LayoutStride
            src_layout = Kokkos.LayoutStride(Base.size_to_strides(1, n...))
        end

        if dst_layout == Kokkos.LayoutStride
            dst_layout = Kokkos.LayoutStride(Base.size_to_strides(1, n...))
        end

        v_src = View{type}(undef, n; mem_space=src_space, layout=src_layout)
        v_dst = View{type}(n; mem_space=dst_space, layout=dst_layout)

        v_src .= rand(type, n) * 10

        if isnothing(exec_space)
            Kokkos.deep_copy(v_dst, v_src)
        else
            Kokkos.deep_copy(exec_space, v_dst, v_src)
            Kokkos.fence(exec_space)
        end

        @test v_dst == v_src
    end
end


@testset "create_mirror to $dst_space_type in $(dim)D" for 
        dst_space_type in (:default_mem_space, Kokkos.COMPILED_MEM_SPACES...),
        dim in Kokkos.COMPILED_DIMS

    dst_mem_space = dst_space_type === :default_mem_space ? nothing : dst_space_type()
    n = ntuple(Returns(7), dim)

    @testset "View{$src_type, $dim, $src_layout, $src_space}" for
            src_type in Kokkos.COMPILED_TYPES,
            src_space in Kokkos.COMPILED_MEM_SPACES,
            src_layout in Kokkos.COMPILED_LAYOUTS

        if src_layout == Kokkos.LayoutStride
            src_layout = Kokkos.LayoutStride(Base.size_to_strides(1, n...))
        end

        v_src = View{src_type}(undef, n; mem_space=src_space, layout=src_layout)
        v_src .= reshape(collect(1:length(v_src)), n)
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


@testset "subview" begin
    v = Kokkos.View{Float64}(undef, 4, 4)
    v[:] .= collect(1:length(v))

    sv1 = Kokkos.subview(v, (2:3, 2:3))
    @test typeof(sv1) === typeof(v)
    @test sv1 == [6.0 10.0 ; 7.0 11.0]

    sv2 = Kokkos.subview(v, (:, 1))
    @test typeof(sv2) === Kokkos.impl_view_type(View{Float64, 1, Kokkos.LayoutStride, Kokkos.HostSpace})
    @test sv2 == [1.0, 2.0, 3.0, 4.0]

    sv3 = Kokkos.subview(v, (1,))
    @test typeof(sv3) === Kokkos.impl_view_type(View{Float64, 1, array_layout(v), memory_space(v)})
    @test sv3 == [1.0, 5.0, 9.0, 13.0]

    sv4 = Kokkos.subview(v, (1, :))
    @test typeof(sv4) === typeof(sv3)
    @test sv3 == sv4

    # Making a (fake) subview from a SubArray, using ranges with non-unit steps 
    sub_v = @view v[1:3:4, 1:3:4]  # Select the "corners" of the matrix
    sv5 = Kokkos.view_wrap(sub_v)  # Uses `strides(sub_v)`
    @test pointer(sv5) == pointer(v)
    @test sv5 == [1.0 13.0 ; 4.0 16.0]
end

end