@testset "Cuda" begin

import Kokkos: Idx, View

@testset "CuArray to View" begin
    A = CuArray{Int64}(undef, 5, 5)
    CUDA.@allowscalar for i in eachindex(A)
        A[i] = i
    end
    A_v = Kokkos.view_wrap(A)

    @test Kokkos.main_view_type(typeof(A_v)) === View{Int64, 2, Kokkos.LayoutLeft, Kokkos.CudaSpace}
    @test UInt(pointer(A_v)) == UInt(pointer(A))
    @test size(A_v) == size(A)
    @test strides(A_v) == strides(A)

    A_vh = Kokkos.create_mirror_view(A_v)
    copyto!(A_vh, A_v)
    @test (@view A_vh[:]) == 1:25

    sub_A = @view A[2:4, 2:4]
    sub_A_v = Kokkos.view_wrap(sub_A)

    @test Kokkos.main_view_type(typeof(sub_A_v)) === View{Int64, 2, Kokkos.LayoutStride, Kokkos.CudaSpace}
    @test UInt(pointer(sub_A_v)) == UInt(pointer(sub_A))
    @test size(sub_A_v) == size(sub_A)
    @test strides(sub_A_v) == strides(sub_A)

    sub_A_vh = Kokkos.create_mirror_view(sub_A_v)
    # sub_A_v and sub_A_vh are not contiguous and therefore cannot be copied directly across devices
    @test !Kokkos.span_is_contiguous(sub_A_v) && !Kokkos.span_is_contiguous(sub_A_vh)
    @test_throws @error_match(r"Kokkos::deep_copy with no available copy mechanism") copyto!(sub_A_vh, sub_A_v)

    sub_A_v_contiguous = Kokkos.View{Int64}(undef, size(sub_A_v); mem_space=Kokkos.CudaSpace, layout=Kokkos.LayoutLeft)
    copyto!(sub_A_v_contiguous, sub_A_v)

    sub_A_vh_contiguous = Kokkos.create_mirror_view(sub_A_v_contiguous)
    copyto!(sub_A_vh_contiguous, sub_A_v_contiguous)

    @test sub_A_vh_contiguous == [7 12 17 ; 8 13 18 ; 9 14 19]
end


@testset "View to CuArray" begin
    V = Kokkos.View{Int64}(undef, 5, 5; mem_space=Kokkos.CudaSpace, layout=Kokkos.LayoutLeft)

    Vh = Kokkos.create_mirror_view(V)
    Vh[:] .= collect(1:25)
    copyto!(V, Vh)

    @test_throws @error_match(r"only possible from the `Kokkos.CudaSpace`") unsafe_wrap(CuArray, Vh)

    cu_V = unsafe_wrap(CuArray, V)
    @test size(cu_V) == size(V)
    @test strides(cu_V) == strides(V)
    CUDA.@allowscalar @test Vh == cu_V

    sub_V = Kokkos.subview(V, (1:3, 1:3))
    @test !Kokkos.span_is_contiguous(sub_V)
    @test_throws @error_match(r"non-contiguous views cannot") unsafe_wrap(CuArray, sub_V)

    sub_V2 = Kokkos.subview(V, (:, 1:3))
    @test Kokkos.span_is_contiguous(sub_V2)
    cu_sub_V2 = unsafe_wrap(CuArray, sub_V2)
    @test size(cu_sub_V2) == size(sub_V2)
    @test strides(cu_sub_V2) == strides(sub_V2)
    CUDA.@allowscalar @test (@view Vh[:, 1:3]) == cu_sub_V2
end

end