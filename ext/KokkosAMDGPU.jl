module KokkosAMDGPU

using Kokkos
import Kokkos: View, view_wrap
isdefined(Base, :get_extension) ? (import AMDGPU) : (import ..AMDGPU)
import AMDGPU: ROCArray


"""
    unsafe_wrap(ROCArray, v::Kokkos.View)

Wrap a `Kokkos.View` into a `ROCArray`. The view must be stored in one of the HIP device memory
spaces.

Views with row-major layout (`LayoutRight`) will be transposed with `Base.adjoint`.

Non-contiguous views (`Kokkos.span_is_contiguous(v) == false`) cannot be represented as `ROCArray`s.

!!! warning

    The returned `ROCArray` does not own the data of the view, which then must stay rooted for the
    entire lifetime of the `ROCArray`.
"""
function Base.unsafe_wrap(
    ::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{ROCArray{T, D}}},
    v::View{T, D, L, S}
) where {T, D, L, S}
    if !(S <: Kokkos.HIPSpace || S <: Kokkos.HIPManagedSpace)
        error("wrapping a Kokkos view into a `ROCArray` is only possible from the \
            `Kokkos.HIPSpace` or `Kokkos.HIPManagedSpace` memory spaces")
    end

    if !Kokkos.span_is_contiguous(v)
        error("non-contiguous (or strided) views cannot be converted into a `ROCArray` \
               (size: $(size(v)), strides: $(strides(v)))")
    end

    device_ptr = Ptr{Cvoid}(pointer(v))

    ptr_info = AMDGPU.Mem.pointerinfo(device_ptr)
    device = get(AMDGPU.Mem.DEVICES, ptr_info.agentOwner.handle, nothing)
    if isnothing(device)
        error("could not retrieve the device of the given pointer: $device_ptr")
    end

    # Manually create a ROCArray using a HIP buffer not managed by AMDGPU
    byte_size = prod(size(v)) * sizeof(T)
    buf = AMDGPU.Mem.HIPBuffer(device, device_ptr, byte_size, false)
    roc_array = ROCArray{T, D}(AMDGPU.DataRef(nothing, buf), dims)  # `nothing` as finalizer: GPUArrays will not attempt to free the array

    if L === Kokkos.LayoutRight
        return roc_array'
    else
        # Either LayoutLeft or contiguous layout (any dimension) => can be represented as a ROCArray
        return roc_array
    end
end


function view_wrap(::Type{View{T, D, L, S}}, a::ROCArray{T, D}; kwargs...) where {T, D, L, S}
    hip_device = AMDGPU.device(a)
    id = AMDGPU.device_id(hip_device)
    kokkos_device_id = Kokkos.BackendFunctions.device_id()
    if id != kokkos_device_id
        error("cannot wrap a view stored in device n°$(id+1) ($hsa_device), since Kokkos is \
               configured to work with the device n°$(kokkos_device_id+1)")
    end
    return view_wrap(View{T, D, L, S}, size(a), pointer(a); kwargs...)
end

view_wrap(::Type{View{T, D}}, a::ROCArray{T, D}; kwargs...) where {T, D} =
    view_wrap(View{T, D, Kokkos.LayoutLeft, Kokkos.HIPSpace}, a; kwargs...)

end
