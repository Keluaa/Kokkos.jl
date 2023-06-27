module KokkosAMDGPU

using Kokkos
import Kokkos: View, view_wrap
isdefined(Base, :get_extension) ? (import AMDGPU) : (import ..AMDGPU)
import AMDGPU


function Base.unsafe_wrap(
    roc_array_t::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{<:ROCArray{T, D}}},
    v::View{T, D, L, S}
) where {T, D, L, S}
    if !(S <: Kokkos.HIPSpace || S <: Kokkos.HIPManagedSpace)
        error("wrapping a Kokkos view into a `ROCArray` is only possible from the \
            `Kokkos.HIPSpace` or `Kokkos.HIPManagedSpace` memory spaces")
    end

    # We assume all Kokkos views are stored in the same device
    kokkos_device_id = Kokkos.Spaces.BackendFunctions.device_id()
    device = AMDGPU.devices(:gpu)[kokkos_device_id + 1]
    # TODO: in tests, if we use ROCm 5.2, there should be a hipDeviceGetUUID (https://github.com/RadeonOpenCompute/ROCm/issues/1642)
    #  => check if AMDGPU & Kokkos share the same ids

    view_ptr = pointer(v)
    lock = false  # We are passing a device pointer directly
    roc_array = Base.unsafe_wrap(roc_array_t, view_ptr, size(v); device, lock)

    if !Kokkos.span_is_contiguous(v)
        error("non-contiguous (or strided) views cannot be converted into a `ROCArray` \
               (size: $(size(v)), strides: $(strides(v)))")
    elseif L === Kokkos.LayoutRight
        return roc_array'
    else
        # Either LayoutLeft or contiguous layout (any dimension) => can be represented as a ROCArray
        return roc_array
    end
end


function view_wrap(::Type{View{T, D, L, S}}, a::ROCArray{T, D}; kwargs...) where {T, D, L, S}
    hsa_device = AMDGPU.device(a)
    id = AMDGPU.device_id(hsa_device) - 1
    kokkos_device_id = Kokkos.Spaces.BackendFunctions.device_id()
    if id != kokkos_device_id
        error("cannot wrap view stored in device n°$(id+1) ($hsa_device), since Kokkos is \
               configured to work with the device n°$(kokkos_device_id+1)")
    end
    return view_wrap(View{T, D, L, S}, size(a), pointer(a) + a.offset; kwargs...)
end

view_wrap(::Type{View{T, D}}, a::ROCArray{T, D}; kwargs...) where {T, D} =
    view_wrap(View{T, D, Kokkos.LayoutLeft, Kokkos.HIP}, size(a), pointer(a) + a.offset; kwargs...)

end
