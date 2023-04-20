@testset "MPI" begin

using MPI

N_PROC = 4
TEST_FILE = joinpath(@__DIR__, "mpi_tests.jl")

@test hasmethod(MPI.Buffer, Tuple{Kokkos.View})

@info "Testing MPI support in $(pwd())"
project = "--project=" * dirname(Base.active_project())
mpi_exec_cmd = mpiexec(; adjust_LIBPATH=false)  # LIBPATH can interfere with CMake
mpi_cmd = `$mpi_exec_cmd -n $N_PROC $(Base.julia_cmd()) $project $TEST_FILE`
mpi_cmd = ignorestatus(mpi_cmd)
p = run(mpi_cmd)
@test success(p)

end
