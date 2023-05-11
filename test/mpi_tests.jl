
using MPI
using Kokkos
using Test

N_PROC = 4

@test hasmethod(MPI.Buffer, Tuple{Kokkos.View})

MPI.Init()

@test MPI.Comm_size(MPI.COMM_WORLD) == N_PROC
rank = MPI.Comm_rank(MPI.COMM_WORLD)

Kokkos.build_in_project()  # Options must be the same on all processes 

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

function test_send_loop(n, v, r)
    prev_rank = (rank + N_PROC - 1) % N_PROC
    next_rank = (rank + 1) % N_PROC

    v[1:end] .= fill_func(n, rank)

    MPI.Sendrecv!(v, r, MPI.COMM_WORLD; dest=next_rank, source=prev_rank)

    @test all(r[1:end] .== fill_func(n, prev_rank))
end

# 1D
n = 10
v = Kokkos.View{Float64}(undef, n)
r = Kokkos.View{Float64}(n)
test_send_loop(n, v, r)

# 2D
n = 100
v = Kokkos.View{Float64}(undef, (10, 10))
r = Kokkos.View{Float64}((10, 10))
test_send_loop(n, v, r)
