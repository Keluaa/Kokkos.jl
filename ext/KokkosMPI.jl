module KokkosMPI

import Kokkos
isdefined(Base, :get_extension) ? (import MPI) : (import ..MPI)
import MPI: MPIPtr


Base.cconvert(::Type{MPIPtr}, v::Kokkos.View) = v
Base.unsafe_convert(::Type{MPIPtr}, v::Kokkos.View) = reinterpret(MPIPtr, pointer(v))


function MPI.Buffer(v::Kokkos.View{T, D}) where {T, D}
    datatype = MPI.Datatype(T)
    count = Cint(1)

    if D == 0
        # `v` can be safely treated like a single `T`
    elseif Kokkos.span_is_contiguous(v)
        # Treat the view as a contiguous block of `T`
        count = Cint(Kokkos.memory_span(v) ÷ sizeof(T))  # equivalent to `Kokkos::size(v)` in this case
    else
        # Build a datatype representing exactly the strided view, stacking each dimension on top of
        # the previous one.
        datatype = MPI.Types.create_vector(size(v, 1), 1, stride(v, 1), datatype)
        for d in 2:D
            datatype = MPI.Types.create_hvector(size(v, d), 1, stride(v, d) * sizeof(T), datatype)
        end
        MPI.Types.commit!(datatype)
    end

    return MPI.Buffer(v, count, datatype)
end

end