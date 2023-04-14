
using Kokkos
using Test


!Kokkos.is_initialized() && Kokkos.initialize()
Kokkos.require(; dims=[1, 2], types=[Float64], exec_spaces=[Kokkos.Serial, Kokkos.OpenMP])


const lib_src = joinpath(@__DIR__, "lib", "simple_lib")
const lib_build = joinpath(Kokkos.KOKKOS_BUILD_DIR, "simple_lib")

const Nx = 100
const Ny = 100

const γ  = 7/5
const ρ₀ = 1
const u₀ = 0
const v₀ = 0
const E₀ = 1


function call_perfect_gas(lib_1D, range, γ, ρ::V, u::V, v::V, E::V, p::V, c::V) where {T, V <: View{T, 1}}
    # void perfect_gas(
    #     Idx start, Idx end,
    #     flt_t gamma,
    #     const view& ρ, const view& u, const view& v, const view& E,
    #     view& p, view& c
    # )
    ccall(Kokkos.get_symbol(lib_1D, :perfect_gas),
        Cvoid, (Idx, Idx, Float64, Ref{V}, Ref{V}, Ref{V}, Ref{V}, Ref{V}, Ref{V}),
        first(range), last(range), γ, ρ, u, v, E, p, c    
    )
end


function call_perfect_gas(lib_2D, range_x, range_y, γ, ρ::V, u::V, v::V, E::V, p::V, c::V) where {T, V <: View{T, 2}}
    # void perfect_gas(
    #     Idx start_x, Idx start_y, Idx end_x, Idx end_y,
    #     flt_t gamma,
    #     const view& ρ, const view& u, const view& v, const view& E,
    #     view& p, view& c
    # )
    ccall(Kokkos.get_symbol(lib_2D, :perfect_gas),
        Cvoid, (Idx, Idx, Idx, Idx, Float64, Ref{V}, Ref{V}, Ref{V}, Ref{V}, Ref{V}, Ref{V}),
        first(range_x), first(range_y), last(range_x), last(range_y), γ, ρ, u, v, E, p, c    
    )
end


function test_lib(lib, dims)
    ρ = View{Float64}(undef, dims)
    u = View{Float64}(undef, dims)
    v = View{Float64}(undef, dims)
    E = View{Float64}(undef, dims)
    p = View{Float64}(undef, dims)
    c = View{Float64}(undef, dims)

    ρ .= ρ₀
    u .= u₀
    v .= v₀
    E .= E₀

    @test all(ρ .== ρ₀)
    @test all(u .== u₀)
    @test all(v .== v₀)
    @test all(E .== E₀)

    if length(dims) == 1
        call_perfect_gas(lib, 1:Nx, γ, ρ, u, v, E, p, c)
    else
        call_perfect_gas(lib, 1:Nx, 1:Ny, γ, ρ, u, v, E, p, c)
    end

    p_expected = (γ - 1) * ρ₀ * (E₀ - (u₀^2 + v₀^2))
    c_expected = √(γ * p_expected / ρ₀);

    @test all(p .≈ p_expected)
    @test all(c .≈ c_expected)
end


project_1D = CMakeKokkosProject(lib_src, "libSimpleKokkosLib1D";
    target="SimpleKokkosLib1D", build_dir=lib_build)
project_2D = CMakeKokkosProject(project_1D, "SimpleKokkosLib2D", "libSimpleKokkosLib2D")


@testset "Simple lib" begin
    @testset "1D" begin
        compile(project_1D)
        lib_1D = Kokkos.load_lib(project_1D)

        @test handle(lib_1D) != C_NULL

        test_lib(lib_1D, Nx)

        Kokkos.unload_lib(lib_1D)
        @test !Kokkos.is_lib_loaded(lib_1D)
    end

    @testset "2D" begin
        compile(project_2D)
        lib_2D = Kokkos.load_lib(project_2D)

        test_lib(lib_2D, (Nx, Ny))

        Kokkos.unload_lib(project_2D)
        @test !Kokkos.is_lib_loaded(project_2D)
    end

    Kokkos.clean(project_1D)  # Also cleans 'project_2D'
    @test isempty(filter!(endswith(".so"), readdir(Kokkos.build_dir(project_1D))))
    Kokkos.clean(project_1D; reset=true)
    @test isempty(readdir(Kokkos.build_dir(project_1D)))
end
