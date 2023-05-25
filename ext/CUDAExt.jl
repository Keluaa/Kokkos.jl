module CUDAExt

import Kokkos: View, view_wrap
isdefined(Base, :get_extension) ? (import CUDA) : (import ..CUDA)
import CUDA: CuPtr, CuArray, AnyCuArray, DenseCuArray, StridedSubCuArray


"""
    unsafe_wrap(CuArray, v::Kokkos.View)

Wrap a `Kokkos.View` into a `CuArray`. The view must be stored in one of the CUDA device memory
spaces.

Views with row-major layout (`LayoutRight`) will be transposed with `Base.adjoint`.

Non-contiguous views (`Kokkos.span_is_contiguous(v) == false`) cannot be represented as `CuArray`s.

!!! warning

    The returned `CuArray` does not own the data of the view, which then must stay rooted for the
    entire lifetime of the `CuArray`.
"""
function Base.unsafe_wrap(
    cu_array_t::Union{Type{CuArray}, Type{CuArray{T}}, Type{<:CuArray{T, D}}},
    v::View{T, D, L, S}
) where {T, D, L, S}
    if !(S <: Kokkos.CudaSpace || S <: Kokkos.CudaUVMSpace)
        error("wrapping a Kokkos view into a `CuArray` is only possible from the \
               `Kokkos.CudaSpace` or `Kokkos.CudaUVMSpace` memory spaces")
    end

    view_ptr = CuPtr{T}(UInt(pointer(v)))
    cu_array = Base.unsafe_wrap(cu_array_t, view_ptr, size(v); own=false)

    if L === Kokkos.LayoutRight
        return cu_array'
    elseif L === Kokkos.LayoutLeft || Kokkos.span_is_contiguous(v)
        return cu_array
    else
        error("Strided non-contiguous views cannot be converted into a `CuArray`")
    end
end


view_wrap(::Type{View{T, D, L, S}}, a::AnyCuArray{T, D}; kwargs...) where {T, D, L, S} =
    view_wrap(View{T, D, L, S}, size(a), pointer(a); kwargs...)

view_wrap(::Type{View{T, D}}, a::DenseCuArray{T, D}) where {T, D} =
    view_wrap(View{T, D, Kokkos.LayoutLeft, Kokkos.CudaSpace}, a)

view_wrap(::Type{View{T, D}}, a::StridedSubCuArray{T, D}) where {T, D} =
    view_wrap(View{T, D, Kokkos.LayoutStride, Kokkos.CudaSpace}, a; layout=LayoutStride(strides(a)))

end