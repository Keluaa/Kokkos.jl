
using Kokkos
using Test

import Kokkos: View, Idx


!Kokkos.is_initialized() && Kokkos.initialize()

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


function call_create_view_by_reference(lib, nx)
    view_t = View{Float64, 1, array_layout(Kokkos.DEFAULT_DEVICE_SPACE), Kokkos.DEFAULT_DEVICE_MEM_SPACE}
    v_ref = view_t()  # Default empty view constructor
    # void create_view(view& v, int nx)
    ccall(Kokkos.get_symbol(lib, :create_view),
        Cvoid,
        # Julia disallows ccall arg types depending on local variables
        # A workaround would be to use a @generated function
        (Ref{View{Float64, 1, array_layout(Kokkos.DEFAULT_DEVICE_SPACE), Kokkos.DEFAULT_DEVICE_MEM_SPACE}}, Cint),
        v_ref, nx
    )
    return v_ref
end


@generated function call_create_view_by_reference(lib, nx, ny)
    view_t = View{Float64, 2, array_layout(Kokkos.DEFAULT_DEVICE_SPACE), Kokkos.DEFAULT_DEVICE_MEM_SPACE}
    return quote
        v_ref = $view_t()  # Default empty view constructor
        # void create_view(view& v, int nx, int ny)
        ccall(Kokkos.get_symbol(lib, :create_view),
            Cvoid,
            (Ref{$view_t}, Cint, Cint),
            v_ref, nx, ny
        )
        return v_ref
    end 
end


function test_lib(lib, dims)
    ρ = View{Float64}(undef, dims)
    u = View{Float64}(undef, dims)
    v = View{Float64}(undef, dims)
    E = View{Float64}(undef, dims)
    p = View{Float64}(undef, dims)
    c = View{Float64}(undef, dims)

    ρ_host = Kokkos.create_mirror_view(ρ)
    u_host = Kokkos.create_mirror_view(u)
    v_host = Kokkos.create_mirror_view(v)
    E_host = Kokkos.create_mirror_view(E)

    ρ_host .= ρ₀
    u_host .= u₀
    v_host .= v₀
    E_host .= E₀

    @test all(ρ_host .== ρ₀)
    @test all(u_host .== u₀)
    @test all(v_host .== v₀)
    @test all(E_host .== E₀)

    copyto!(ρ, ρ_host)
    copyto!(u, u_host)
    copyto!(v, v_host)
    copyto!(E, E_host)

    if length(dims) == 1
        call_perfect_gas(lib, 1:Nx, γ, ρ, u, v, E, p, c)
    else
        call_perfect_gas(lib, 1:Nx, 1:Ny, γ, ρ, u, v, E, p, c)
    end

    p_expected = (γ - 1) * ρ₀ * (E₀ - (u₀^2 + v₀^2))
    c_expected = √(γ * p_expected / ρ₀)

    p_host = Kokkos.create_mirror_view(p)
    c_host = Kokkos.create_mirror_view(c)

    copyto!(p_host, p)
    copyto!(c_host, c)

    @test all(p_host .≈ p_expected)
    @test all(c_host .≈ c_expected)

    if length(dims) == 1
        created_view = call_create_view_by_reference(lib, first(dims))
    else
        created_view = call_create_view_by_reference(lib, first(dims), last(dims))
    end
    @test created_view.cpp_object != C_NULL
    @test size(created_view) == dims
    TEST_DEVICE_ACCESSIBLE && @test all(created_view .== 0)
    @test label(created_view) == (length(dims) == 1 ? "test_ref_1D" : "test_ref_2D")

    # Very important: finalize the view manually, since we unload the library right after, which
    # contains the view deallocation function. Not doing so will segfault at the next GC pass (or
    # when Julia calls all finalizers before exiting).
    finalize(created_view)
end


project_1D = CMakeKokkosProject(lib_src, "libSimpleKokkosLib1D";
    target="SimpleKokkosLib1D", build_dir=lib_build)
project_2D = CMakeKokkosProject(project_1D, "SimpleKokkosLib2D", "libSimpleKokkosLib2D")

if isdir(Kokkos.build_dir(project_1D))
    Kokkos.clean(project_1D)
end


@testset "Simple lib" begin
    @testset "1D" begin
        compile(project_1D; loading_bar=true)
        lib_1D = Kokkos.load_lib(project_1D)

        @test handle(lib_1D) != C_NULL

        test_lib(lib_1D, (Nx,))

        Kokkos.unload_lib(lib_1D)
    end

    @testset "2D" begin
        compile(project_2D; loading_bar=true)
        lib_2D = Kokkos.load_lib(project_2D)

        test_lib(lib_2D, (Nx, Ny))

        Kokkos.unload_lib(project_2D)
    end

    Kokkos.clean(project_1D)  # Also cleans 'project_2D'
    @test isempty(filter!(endswith(".so"), readdir(Kokkos.build_dir(project_1D))))
    Kokkos.clean(project_1D; reset=true)
    @test isempty(readdir(Kokkos.build_dir(project_1D)))
end
