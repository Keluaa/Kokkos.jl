@testset "Views (inaccessible)" begin
# Same as 'test/views.jl', but with all view accesses removed, as they are inaccessible from the host.

TEST_VIEW_MEM_SPACES = if TEST_CUDA
    (Kokkos.CudaSpace, Kokkos.CudaUVMSpace)
else
    (Kokkos.HIPSpace, Kokkos.HIPManagedSpace)
end

TEST_DEFAULT_VIEW_TYPE = if TEST_CUDA
    Kokkos.KokkosWrapper.Impl.View1D_L_CudaAllocated
else
    Kokkos.KokkosWrapper.Impl.View1D_L_HIPAllocated
end


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
@test v1 isa TEST_DEFAULT_VIEW_TYPE{Float64}
@test v1 isa AbstractArray
@test v1 isa AbstractArray{Float64}
@test v1 isa AbstractArray{Float64, 1}

@test !accessible(v1)
@test label(v1) == ""

@test size(v1) == (n1,)
@test length(v1) == n1
@test ndims(v1) == 1
@test strides(v1) == (1,)
@test Base.elsize(v1) == sizeof(Float64)
@test memory_space(v1) in TEST_VIEW_MEM_SPACES

v2 = View{Float64}(undef, n1; label="v2")
@test label(v2) == "v2"

n3 = (n1, n1)
v3 = View{Int64, 2, Kokkos.LayoutRight}(n3)

@test size(v3) == n3
@test length(v3) == prod(n3)
@test ndims(v3) == 2
@test strides(v3) == (n1, 1)
@test Base.elsize(v3) == sizeof(Float64)
@test sizeof(v3) == sizeof(Float64) * prod(n3)
@test memory_space(v3) in TEST_VIEW_MEM_SPACES

@testset "LayoutLeft" begin
    v3_l = View{Int64, 2, Kokkos.LayoutLeft}(n3)

    @test size(v3_l) == n3
    @test length(v3_l) == prod(n3)
    @test ndims(v3_l) == 2
    @test strides(v3_l) == (1, n1)
    @test Base.elsize(v3_l) == sizeof(Float64)
    @test sizeof(v3_l) == sizeof(Float64) * prod(n3)
    @test memory_space(v3_l) in TEST_VIEW_MEM_SPACES
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
    @test memory_space(v3_s) in TEST_VIEW_MEM_SPACES
end

v4 = similar(v3)
@test size(v4) == size(v3)
@test typeof(v4) == typeof(v3)
@test strides(v4) == strides(v3)
@test memory_space(v4) == memory_space(v3)

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
    View{Float64}(1, 2; mem_space=TEST_MEM_SPACE),
    View{Float64}(1, 2; mem_space=TEST_MEM_SPACE()),
    View{Float64}(1, 2; mem_space=TEST_MEM_SPACE, layout=Kokkos.LayoutRight),
    View{Float64}(1, 2; mem_space=TEST_MEM_SPACE, layout=Kokkos.LayoutRight())
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
@test_throws @error_match("$(nameof(TEST_UNAVAILABLE_MEM_SPACE)) is not compiled") View{Int64}(undef, n1; mem_space=TEST_UNAVAILABLE_MEM_SPACE)
@test_throws @error_match("`mem_space` kwarg") View{Float64, 1, Kokkos.LayoutLeft, Kokkos.KokkosWrapper.Impl.HostSpaceImplDereferenced}(undef, n1; mem_space=Kokkos.HostSpace)
@test_throws @error_match("Kokkos.Views.LayoutLeft type") View{Float64, 1, Kokkos.LayoutLeft}(undef, n1; layout=Kokkos.LayoutRight)
@test_throws @error_match("requires a instance") View{Float64}(undef, n1; layout=Kokkos.LayoutStride)


# TODO: view_wrap test, but for each backend, from the respective packages CUDA.jl and AMDGPU.jl

@testset "Deep copy on $exec_space_type in $(dim)D with $type" for 
        exec_space_type in (:no_exec_space, Kokkos.COMPILED_EXEC_SPACES...),
        dim in Kokkos.COMPILED_DIMS,
        type in Kokkos.COMPILED_TYPES

    exec_space = exec_space_type === :no_exec_space ? nothing : exec_space_type()
    n = ntuple(Returns(7), dim)

    @testset "View{$type, $dim, $src_layout, $src_space} => View{$type, $dim, $dst_layout, $dst_space}" for
            src_space in Kokkos.COMPILED_MEM_SPACES, dst_space in Kokkos.COMPILED_MEM_SPACES,
            src_layout in Kokkos.COMPILED_LAYOUTS, dst_layout in Kokkos.COMPILED_LAYOUTS

        src_view_t = View{type, dim, src_layout, src_space}
        dst_view_t = View{type, dim, dst_layout, dst_space}
        if !((src_layout == dst_layout) || exec_space === :no_exec_space ||
                (accessible(exec_space_type, src_space) && accessible(exec_space, dst_scapce)))
            # As per the Kokkos::deep_copy docs, there should not be a valid deep_copy method
            # See https://kokkos.github.io/kokkos-core-wiki/API/core/view/deep_copy.html#requirements
            if exec_space === :no_exec_space
                @test isempty(methods(Kokkos.deep_copy, (src_view_t, dst_view_t)))
            else
                @test isempty(methods(Kokkos.deep_copy, (exec_space_type, src_view_t, dst_view_t)))
            end
            continue
        else
            if exec_space === :no_exec_space
                @test !isempty(methods(Kokkos.deep_copy, (src_view_t, dst_view_t)))
            else
                @test !isempty(methods(Kokkos.deep_copy, (exec_space_type, src_view_t, dst_view_t)))
            end
        end

        if src_layout == Kokkos.LayoutStride
            src_layout = Kokkos.LayoutStride(Base.size_to_strides(1, n...))
        end

        if dst_layout == Kokkos.LayoutStride
            dst_layout = Kokkos.LayoutStride(Base.size_to_strides(1, n...))
        end

        v_src = View{type}(undef, n; mem_space=src_space, layout=src_layout)
        v_dst = View{type}(n; mem_space=dst_space, layout=dst_layout)

        if isnothing(exec_space)
            Kokkos.deep_copy(v_dst, v_src)
        else
            Kokkos.deep_copy(exec_space, v_dst, v_src)
            Kokkos.fence(exec_space)
        end
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
        v_src_m = Kokkos.create_mirror(v_src; mem_space=dst_mem_space, zero_fill=true)

        if isnothing(dst_mem_space)
            @test Kokkos.accessible(Kokkos.memory_space(v_src_m))
        else
            @test Kokkos.memory_space(v_src_m) == Kokkos.main_space_type(dst_mem_space)
        end
    end
end

end