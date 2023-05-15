
#include <Kokkos_Core.hpp>


using flt_t = double;
using RangePolicy = typename Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
using Idx = RangePolicy::index_type;
using view = Kokkos::View<flt_t**>;


KOKKOS_INLINE_FUNCTION flt_t kinetic_energy(flt_t u, flt_t v)
{
    return flt_t(0.5) * (std::pow(u, flt_t(2)) + std::pow(v, flt_t(2)));
}


KOKKOS_INLINE_FUNCTION flt_t internal_energy(flt_t E, flt_t Ec)
{
    return E - Ec;
}


extern "C"
void perfect_gas(
        Idx start_x, Idx start_y, Idx end_x, Idx end_y,
        flt_t gamma,
        const view& r, const view& u, const view& v, const view& E,
        view& p, view& c
    )
{
    auto array_2D_range = RangePolicy({start_x - 1, start_y - 1}, {end_x, end_y});
    Kokkos::parallel_for(array_2D_range,
    KOKKOS_LAMBDA(const Idx i, const Idx j) {
        flt_t e = internal_energy(E(i, j), kinetic_energy(u(i, j), v(i, j)));
        p(i, j) = (gamma - 1) * r(i, j) * e;
        c(i, j) = std::sqrt(gamma * p(i, j) / r(i, j));
    });
}


extern "C"
void create_view(view& v, int nx, int ny)
{
    v = view("test_ref_2D", nx, ny);
}
