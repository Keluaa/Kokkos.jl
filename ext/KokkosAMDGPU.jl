module KokkosAMDGPU

using Kokkos
import Kokkos: View, view_wrap
isdefined(Base, :get_extension) ? (import AMDGPU) : (import ..AMDGPU)
import AMDGPU: ROCArray


const AMDGPU_VERSION = @static if VERSION ≥ v"1.9-"
    pkgversion(AMDGPU)
elseif VERSION ≥ v"1.8-"
    # This is just a reimplementation of `pkgversion`, which returns v"0.0.1" by default
    pkg = Base.PkgId(AMDGPU)
    origin = get(Base.pkgorigins, pkg, nothing)
    isnothing(origin) ? v"0.0.1" : origin.version
else
    v"0.0.1"  # No `version` field before 1.8
end


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
    roc_array_t::Union{Type{ROCArray}, Type{ROCArray{T}}, Type{<:ROCArray{T, D}}},
    v::View{T, D, L, S}
) where {T, D, L, S}
    if !(S <: Kokkos.HIPSpace || S <: Kokkos.HIPManagedSpace)
        error("wrapping a Kokkos view into a `ROCArray` is only possible from the \
            `Kokkos.HIPSpace` or `Kokkos.HIPManagedSpace` memory spaces")
    end

    # We assume all Kokkos views are stored in the same device
    kokkos_device_id = Kokkos.BackendFunctions.device_id()
    device = AMDGPU.devices(:gpu)[kokkos_device_id + 1]
    # TODO: in tests, if we use ROCm 5.2, there should be a hipDeviceGetUUID (https://github.com/RadeonOpenCompute/ROCm/issues/1642)
    #  => check if AMDGPU & Kokkos share the same ids

    view_ptr = pointer(v)
    lock = false  # We are passing a device pointer directly

    roc_array = @static if AMDGPU_VERSION ≥ v"0.4.16"  # TODO: check if the bug is corrected in that version
        Base.unsafe_wrap(roc_array_t, view_ptr, size(v); device, lock)
    else
        # Workaround for https://github.com/JuliaGPU/AMDGPU.jl/issues/436
        # Reimplementation of AMDGPU.unsafe_wrap with correct pointer type conversion
        sz = prod(size(v)) * sizeof(T)
        device_ptr = Ptr{Cvoid}(view_ptr)
        buf = AMDGPU.Mem.Buffer(device_ptr, Ptr{Cvoid}(view_ptr), device_ptr, sz, device, false, false)
        ROCArray{T, D}(buf, size(v))
    end

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
    kokkos_device_id = Kokkos.BackendFunctions.device_id()
    if id != kokkos_device_id
        error("cannot wrap view stored in device n°$(id+1) ($hsa_device), since Kokkos is \
               configured to work with the device n°$(kokkos_device_id+1)")
    end
    return view_wrap(View{T, D, L, S}, size(a), pointer(a); kwargs...)
end

view_wrap(::Type{View{T, D}}, a::ROCArray{T, D}; kwargs...) where {T, D} =
    view_wrap(View{T, D, Kokkos.LayoutLeft, Kokkos.HIPSpace}, a; kwargs...)

end