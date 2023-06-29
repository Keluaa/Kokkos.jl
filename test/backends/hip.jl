@testset "HIP" begin

import Kokkos: View

@testset "ROCArray to View" begin
    A = ROCArray{Int64}(undef, 5, 5)
    AMDGPU.@allowscalar for i in eachindex(A)
        A[i] = i
    end
    A_v = Kokkos.view_wrap(A)

    @test Kokkos.main_view_type(typeof(A_v)) === View{Int64, 2, Kokkos.LayoutLeft, Kokkos.HIPSpace}
    @test UInt(pointer(A_v)) == UInt(pointer(A))
    @test size(A_v) == size(A)
    @test strides(A_v) == strides(A)

    A_vh = Kokkos.create_mirror_view(A_v)
    copyto!(A_vh, A_v)
    @test (@view A_vh[:]) == 1:25

    sub_A = @view A[5:end]  # Returns a contiguous ROCArray with a non-zero offset to its parent
    @test sub_A isa ROCArray
    @test pointer(sub_A) == pointer(A) + sub_A.offset

    sub_A_v = Kokkos.view_wrap(sub_A)

    @test Kokkos.main_view_type(typeof(sub_A_v)) === View{Int64, 1, Kokkos.LayoutLeft, Kokkos.HIPSpace}
    @test UInt(pointer(sub_A_v)) == UInt(pointer(sub_A))
    @test size(sub_A_v) == size(sub_A)
    @test strides(sub_A_v) == strides(sub_A)

    sub_A_vh = Kokkos.create_mirror_view(sub_A_v)
    copyto!(sub_A_vh, sub_A_v)

    @test sub_A_vh == 5:25
end


@testset "View to ROCArray" begin
    V = Kokkos.View{Int64}(undef, 5, 5; mem_space=Kokkos.HIPSpace, layout=Kokkos.LayoutLeft)

    Vh = Kokkos.create_mirror_view(V)
    Vh[:] .= collect(1:25)
    copyto!(V, Vh)

    @test_throws @error_match(r"only possible from the `Kokkos.HIPSpace`") unsafe_wrap(ROCArray, Vh)

    roc_V = unsafe_wrap(ROCArray, V)
    @test size(roc_V) == size(V)
    @test strides(roc_V) == strides(V)
    AMDGPU.@allowscalar @test Vh == roc_V

    sub_V = Kokkos.subview(V, (1:3, 1:3))
    @test !Kokkos.span_is_contiguous(sub_V)
    @test_throws @error_match(r"non-contiguous \(or strided\) views cannot") unsafe_wrap(ROCArray, sub_V)

    sub_V2 = Kokkos.subview(V, (:, 1:3))
    @test Kokkos.span_is_contiguous(sub_V2)
    roc_sub_V2 = unsafe_wrap(ROCArray, sub_V2)
    @test size(roc_sub_V2) == size(sub_V2)
    @test strides(roc_sub_V2) == strides(sub_V2)
    AMDGPU.@allowscalar @test (@view Vh[:, 1:3]) == roc_sub_V2
end


@testset "Backend Functions" begin
    BF = Kokkos.BackendFunctions
    exec = Kokkos.HIP()

    did = AMDGPU.default_device_id() - 1

    @test BF.device_id() == did
    @test BF.device_id(exec) == did

    if Kokkos.KOKKOS_VERSION > v"4.0-"
        @test BF.stream_ptr(exec) != C_NULL
    else
        # In v3, the global HIPInstance has a null stream
        @test BF.stream_ptr(exec) == C_NULL
    end

    stream = BF.stream_ptr(exec)
    wrapped_exec = BF.wrap_stream(stream)
    @test typeof(wrapped_exec) === typeof(exec)
    @test BF.device_id(wrapped_exec) == did
end

end
