module MPIExt

# Follows the same structure of the extensions for CUDA.jl and AMDGPU.jl in MPI.jl

import Kokkos
isdefined(Base, :get_extension) ? (import MPI) : (import ..MPI)
import MPI: MPIPtr, Buffer, Datatype


Base.cconvert(::Type{MPIPtr}, arr::Kokkos.View) = arr

function Base.unsafe_convert(::Type{MPIPtr}, arr::Kokkos.View{T}) where T
    return reinterpret(MPIPtr, pointer(arr))
end

function Buffer(arr::Kokkos.View)
    # TODO: support for complex view layouts
    if Kokkos.memory_span(arr) % sizeof(eltype(arr)) != 0
        error("the view's memory span is irregular. \n\
               MPI support for complex view layouts (or with padding) is not yet implemented.")
    end
    return Buffer(arr, Cint(Kokkos.memory_span(arr) ÷ sizeof(eltype(arr))), Datatype(eltype(arr)))
end

end