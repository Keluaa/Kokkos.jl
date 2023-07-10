
using MPI
using Kokkos
using Test

N_PROC = 4

@test hasmethod(MPI.Buffer, Tuple{Kokkos.View})

MPI.Init()

@test MPI.Comm_size(MPI.COMM_WORLD) == N_PROC
rank = MPI.Comm_rank(MPI.COMM_WORLD)

Test.TESTSET_PRINT_ENABLE[] = rank == 0  # Print results only on root

# Important: make sure that all compilation occurs on the root process
rank == 0 && Kokkos.load_wrapper_lib()

MPI.Barrier(MPI.COMM_WORLD)

rank != 0 && Kokkos.load_wrapper_lib(; no_compilation=true, no_git=true)
Kokkos.initialize()


fill_func(n, r) = (1:n) .+ (n * 10 * r)

function test_send_loop(s, mem_space)
    v = Kokkos.View{Float64}(undef, s; mem_space)
    r = Kokkos.View{Float64}(s; mem_space)

    n = length(v)

    v_host = Kokkos.create_mirror_view(v)
    v_host[1:end] .= fill_func(n, rank)
    copyto!(v, v_host)

    prev_rank = (rank + N_PROC - 1) % N_PROC
    next_rank = (rank + 1) % N_PROC
    MPI.Sendrecv!(v, r, MPI.COMM_WORLD; dest=next_rank, source=prev_rank)

    r_host = Kokkos.create_mirror_view(r)
    copyto!(r_host, r)
    @test all(r_host[1:end] .== fill_func(n, prev_rank))
end


function test_send_subview_loop(s, i, mem_space)
    v = Kokkos.View{Float64}(s; mem_space)
    r = Kokkos.View{Float64}(s; mem_space)

    n = length(@view v[i...])

    v_host = Kokkos.create_mirror_view(v)
    (@views v_host[i...][:]) .= fill_func(n, rank)
    copyto!(v, v_host)

    v_sub = Kokkos.subview(v, i)
    r_sub = Kokkos.subview(r, i)

    prev_rank = (rank + N_PROC - 1) % N_PROC
    next_rank = (rank + 1) % N_PROC
    MPI.Sendrecv!(v_sub, r_sub, MPI.COMM_WORLD; dest=next_rank, source=prev_rank)

    r_host = Kokkos.create_mirror_view(r)
    copyto!(r_host, r)
    @test all((@views r_host[i...][:]) .== fill_func(n, prev_rank))
end


@testset "MPI" begin
    @testset "1D" begin
        s = 10
        test_send_loop(s, Kokkos.DEFAULT_HOST_MEM_SPACE)  # Host to Host
        test_send_loop(s, Kokkos.DEFAULT_DEVICE_MEM_SPACE)  # Device to Device 
    end

    @testset "2D" begin
        s = (10, 10)
        test_send_loop(s, Kokkos.DEFAULT_HOST_MEM_SPACE)  # Host to Host
        test_send_loop(s, Kokkos.DEFAULT_DEVICE_MEM_SPACE)  # Device to Device 
    end

    @testset "1D - Strided" begin
        s = (10, 9)
        i = (3:8, 7)  # 1D view of a 2D array
        test_send_subview_loop(s, i, Kokkos.DEFAULT_HOST_MEM_SPACE)  # Host to Host
        test_send_subview_loop(s, i, Kokkos.DEFAULT_DEVICE_MEM_SPACE)  # Device to Device 
    end

    @testset "2D - Strided" begin
        s = (10, 9)
        i = (3:8, 4:9)
        test_send_subview_loop(s, i, Kokkos.DEFAULT_HOST_MEM_SPACE)  # Host to Host
        test_send_subview_loop(s, i, Kokkos.DEFAULT_DEVICE_MEM_SPACE)  # Device to Device  
    end

    GC.gc(true)
    @test_nowarn Kokkos.finalize()
end
