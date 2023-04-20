
cd(joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

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
