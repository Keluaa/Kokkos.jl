
#include <Kokkos_Core.hpp>


using flt_t = double;
using RangePolicy = typename Kokkos::RangePolicy<>;
using Idx = RangePolicy::index_type;
using view = Kokkos::View<flt_t*>;


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
        Idx start, Idx end,
        flt_t gamma,
        const view& ρ, const view& u, const view& v, const view& E,
        view& p, view& c
    )
{
    // Julia is 1-indexed, C++ is 0-indexed, therefore we need to subtract 1 to convert the indexes.
    // Kokkos excludes the end index, while Julia includes it, therefore we add 1 to the end index.
    auto array_range = RangePolicy(start - 1, end);
    Kokkos::parallel_for(array_range,
    KOKKOS_LAMBDA(const Idx i) {
        flt_t e = internal_energy(E[i], kinetic_energy(u[i], v[i]));
        p[i] = (gamma - 1) * ρ[i] * e;
        c[i] = std::sqrt(gamma * p[i] / ρ[i]);
    });
}
