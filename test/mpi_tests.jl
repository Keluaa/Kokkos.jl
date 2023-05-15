
using MPI
using Kokkos
using Test

N_PROC = 4

@test hasmethod(MPI.Buffer, Tuple{Kokkos.View})

MPI.Init()

@test MPI.Comm_size(MPI.COMM_WORLD) == N_PROC
rank = MPI.Comm_rank(MPI.COMM_WORLD)


# Important: make sure that all compilation occurs on the root process
if rank == 0
    invalid_config = !Kokkos.require(;
        dims=[1, 2],
        types=[Float64],
        exec_spaces=[Kokkos.Serial],
        no_error=true
    )
    if invalid_config
        @warn "Invalid Kokkos configuration"
        Kokkos.configinfo()
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
    Kokkos.load_wrapper_lib()
end

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


# 1D
s = 10
test_send_loop(s, Kokkos.DEFAULT_HOST_MEM_SPACE)  # Host to Host
test_send_loop(s, Kokkos.DEFAULT_DEVICE_MEM_SPACE)  # Device to Device

# 2D
s = (10, 10)
test_send_loop(s, Kokkos.DEFAULT_HOST_MEM_SPACE)  # Host to Host
test_send_loop(s, Kokkos.DEFAULT_DEVICE_MEM_SPACE)  # Device to Device
