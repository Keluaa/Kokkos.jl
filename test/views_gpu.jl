@testset "Views (inaccessible)" begin
# Same as 'test/views.jl', but with all view accesses removed, as they are inaccessible from the host.

TEST_VIEW_MEM_SPACES = if TEST_CUDA
    (Kokkos.CudaSpace, Kokkos.CudaUVMSpace)
else
    (Kokkos.HIPSpace, Kokkos.HIPManagedSpace)
end

TEST_DEFAULT_VIEW_TYPE = Symbol("View1D_L_", nameof(TEST_BACKEND_DEVICE), "Allocated")
TEST_DEFAULT_DEVICE_LAYOUT = array_layout(TEST_BACKEND_DEVICE)


import Kokkos: Idx, View

@test Idx <: Union{UInt64, Int64, UInt32, Int32}

@test View <: AbstractArray
@test View{Float64} <: AbstractArray{Float64}
@test View{Float64, 2} <: AbstractArray{Float64, 2}
@test View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT} <: AbstractArray{Float64, 2}
@test View{Float64, 3, TEST_DEFAULT_DEVICE_LAYOUT, TEST_MAIN_MEM_SPACE_DEVICE} <: AbstractArray{Float64, 3}

n1 = 10
v1 = View{Float64}(n1)

@test v1 isa View
@test v1 isa View{Float64}
@test v1 isa View{Float64, 1}
@test v1 isa View{Float64, 1, <:Kokkos.Layout}
@test v1 isa View{Float64, 1, TEST_DEFAULT_DEVICE_LAYOUT}
@test v1 isa View{Float64, 1, TEST_DEFAULT_DEVICE_LAYOUT, <:TEST_MAIN_MEM_SPACE_DEVICE}
@test nameof(typeof(v1)) === TEST_DEFAULT_VIEW_TYPE
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

flat_v4 = @view v4[:]
@test length(flat_v4) == length(v4)

# View constructors
v6 = View{Float64, 2}(undef, (1, 2))
v6_simili = [
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT, TEST_MAIN_MEM_SPACE_DEVICE}(undef, (1, 2)),
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT, TEST_MAIN_MEM_SPACE_DEVICE}(undef, 1, 2),
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT, TEST_MAIN_MEM_SPACE_DEVICE}((1, 2)),
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT, TEST_MAIN_MEM_SPACE_DEVICE}(1, 2),
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT}(undef, (1, 2)),
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT}(undef, 1, 2),
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT}((1, 2)),
    View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT}(1, 2),
    View{Float64, 2}(undef, (1, 2); layout=TEST_DEFAULT_DEVICE_LAYOUT),
    View{Float64, 2}(undef, 1, 2; layout=TEST_DEFAULT_DEVICE_LAYOUT),
    View{Float64, 2}((1, 2); layout=TEST_DEFAULT_DEVICE_LAYOUT),
    View{Float64, 2}(1, 2; layout=TEST_DEFAULT_DEVICE_LAYOUT),
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
    View{Float64}(1, 2; mem_space=TEST_MAIN_MEM_SPACE_DEVICE),
    View{Float64}(1, 2; mem_space=TEST_MAIN_MEM_SPACE_DEVICE()),
    View{Float64}(1, 2; mem_space=TEST_MAIN_MEM_SPACE_DEVICE, layout=TEST_DEFAULT_DEVICE_LAYOUT),
    View{Float64}(1, 2; mem_space=TEST_MAIN_MEM_SPACE_DEVICE, layout=TEST_DEFAULT_DEVICE_LAYOUT())
]
for v6_s in v6_simili
    @test typeof(v6) == typeof(v6_s)
    @test size(v6) == size(v6_s)
    @test strides(v6) == strides(v6_s)
end

@test size(View{Float64}()) == (0,)
@test size(View{Float64, 2}()) == (0, 0)
@test size(View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT}()) == (0, 0)
@test size(View{Float64, 2, TEST_DEFAULT_DEVICE_LAYOUT, TEST_MAIN_MEM_SPACE_DEVICE}()) == (0, 0)

@test_throws @error_match("$(nameof(TEST_UNAVAILABLE_MEM_SPACE)) is not compiled") View{Int64}(undef, n1; mem_space=TEST_UNAVAILABLE_MEM_SPACE)
@test_throws @error_match("`mem_space` kwarg") View{Float64, 1, Kokkos.LayoutLeft, Kokkos.Wrapper.Impl.HostSpaceImplDereferenced}(undef, n1; mem_space=Kokkos.HostSpace)
@test_throws @error_match("Kokkos.LayoutLeft type") View{Float64, 1, Kokkos.LayoutLeft}(undef, n1; layout=Kokkos.LayoutRight)
@test_throws @error_match("requires a instance") View{Float64}(undef, n1; layout=Kokkos.LayoutStride)


# LayoutStride, but with a contiguous row-major layout
v9 = Kokkos.View{Float64}(undef, 3, 4; layout=Kokkos.LayoutStride(Base.size_to_strides(1, 3, 4)))
@test size(v9) == (3, 4)
@test strides(v9) == Base.size_to_strides(1, 3, 4)
@test Kokkos.span_is_contiguous(v9)

# LayoutStride, but with a non-contiguous layout (each element takes twice as much space)
v10 = Kokkos.View{Float64}(undef, 3, 4; layout=Kokkos.LayoutStride(Base.size_to_strides(2, 3, 4)))
@test size(v10) == (3, 4)
@test strides(v10) == Base.size_to_strides(2, 3, 4)
@test !Kokkos.span_is_contiguous(v10)
@test Kokkos.memory_span(v10) == prod(size(v10)) * sizeof(Float64) * 2 


@testset "Deep copy on $exec_space_type in $(dim)D with $type" for 
        exec_space_type in (:no_exec_space, Kokkos.ENABLED_EXEC_SPACES...),
        dim in TEST_VIEW_DIMS,
        type in TEST_VIEW_TYPES

    exec_space = exec_space_type === :no_exec_space ? nothing : exec_space_type()
    n = ntuple(Returns(7), dim)

    @testset "View{$type, $dim, $src_layout, $src_space} => View{$type, $dim, $dst_layout, $dst_space}" for
            src_space in Kokkos.ENABLED_MEM_SPACES, dst_space in Kokkos.ENABLED_MEM_SPACES,
            src_layout in TEST_VIEW_LAYOUTS,
            dst_layout in TEST_VIEW_LAYOUTS

        src_view_t = Kokkos.Views.impl_view_type(View{type, dim, src_layout, src_space})
        dst_view_t = Kokkos.Views.impl_view_type(View{type, dim, dst_layout, dst_space})
        if !((src_layout == dst_layout) || exec_space === :no_exec_space ||
                (accessible(exec_space_type, src_space) && accessible(exec_space, dst_space)))
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
        dst_space_type in (:default_mem_space, Kokkos.ENABLED_MEM_SPACES...),
        dim in TEST_VIEW_DIMS

    dst_mem_space = dst_space_type === :default_mem_space ? nothing : dst_space_type()
    n = ntuple(Returns(7), dim)

    @testset "View{$src_type, $dim, $src_layout, $src_space}" for
            src_type in TEST_VIEW_TYPES,
            src_space in Kokkos.ENABLED_MEM_SPACES,
            src_layout in TEST_VIEW_LAYOUTS

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


@testset "subview" begin
    v = Kokkos.View{Float64}(undef, 4, 4)

    sv1 = Kokkos.subview(v, (2:3, 2:3))
    @test typeof(sv1) === typeof(v)

    sv2 = Kokkos.subview(v, (:, 1))
    @test typeof(sv2) === Kokkos.impl_view_type(View{Float64, 1, array_layout(v), Kokkos.CudaSpace})
    @test Kokkos.main_view_type(sv2) === View{Float64, 1, array_layout(v), Kokkos.CudaSpace}

    sv3 = Kokkos.subview(v, (1,))
    @test typeof(sv3) === Kokkos.impl_view_type(View{Float64, 1, Kokkos.LayoutStride, memory_space(v)})
    @test Kokkos.main_view_type(sv3) === View{Float64, 1, Kokkos.LayoutStride, memory_space(v)}

    sv4 = Kokkos.subview(v, (1, :))
    @test typeof(sv4) === typeof(sv3)
end

end
