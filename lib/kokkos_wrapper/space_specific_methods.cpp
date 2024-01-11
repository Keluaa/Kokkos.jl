
#include "spaces.h"
#include "execution_spaces.h"
#include "kokkos_utils.h"


void import_backend_methods(jl_module_t* impl_module, jl_module_t* backend_funcs_module, const std::vector<const char*>& methods)
{
    // In order to override the methods in the Kokkos.BackendFunctions module, we must have them imported
    for (auto& method : methods) {
        jl_module_import(impl_module, backend_funcs_module, jl_symbol(method));
    }
}


template<typename>
void space_methods(jlcxx::Module&, jl_module_t*) {}


#ifdef KOKKOS_ENABLE_OPENMP
#include <omp.h>

// Weak symbol, for compatibility with versions of OpenMP which do not define this function
#pragma weak omp_capture_affinity
extern "C" size_t omp_capture_affinity(char *buffer, size_t size, const char *format); // NOLINT(*-redundant-declaration)


jl_value_t* capture_affinity(const char* format)
{
    if (!omp_capture_affinity) {
        jl_error("'omp_capture_affinity' is not defined in this version of OpenMP");
    }

    std::vector<std::string> affinities(omp_get_max_threads());

#pragma omp parallel for ordered default(none) shared(format, affinities, std::cout)
    for (int i = 0; i < omp_get_max_threads(); i++) {
        size_t output_size = omp_capture_affinity(nullptr, 0, format);
        std::vector<char> raw_output(output_size + 1);
        omp_capture_affinity(raw_output.data(), raw_output.size(), format);

        std::stringstream ss;
        ss << "thread_num=" << omp_get_thread_num()
           << ", thread_affinity=" << std::string(raw_output.begin(), raw_output.end() - 1) << "\n";
        affinities[omp_get_thread_num()] = ss.str();
    }

    std::stringstream merged;
    for (const auto& affinity : affinities) {
        merged << affinity;
    }

    return jl_cstr_to_string(merged.str().c_str());
}


template<>
void space_methods<SpaceInfo<Kokkos::OpenMP>>(jlcxx::Module& mod, jl_module_t* backend_funcs_module)
{
    import_backend_methods(mod.julia_module(), backend_funcs_module, {
        "omp_set_num_threads",
        "omp_get_max_threads",
        "omp_get_proc_bind",
        "omp_get_num_places",
        "omp_get_place_num_procs",
        "omp_get_place_proc_ids",
        "omp_capture_affinity"
    });

    mod.method("omp_set_num_threads", [](int num){ omp_set_num_threads(num); });
    mod.method("omp_get_max_threads", [](){ return omp_get_max_threads(); });
    mod.method("omp_get_proc_bind", [](){ return static_cast<int>(omp_get_proc_bind()); });
    mod.method("omp_get_num_places", [](){ return omp_get_num_places(); });
    mod.method("omp_get_place_num_procs", [](int place){ return omp_get_place_num_procs(place); });
    mod.method("omp_get_place_proc_ids", [](int place) {
        int array_size = omp_get_place_num_procs(place);
        std::vector<int> ids(array_size);
        omp_get_place_proc_ids(place, ids.data());
        return ids;
    });
    mod.method("omp_capture_affinity", capture_affinity);
    mod.method("omp_capture_affinity", [](){ return capture_affinity(nullptr); });
}
#endif


#ifdef KOKKOS_ENABLE_CUDA
template<>
void space_methods<SpaceInfo<Kokkos::Cuda>>(jlcxx::Module& mod, jl_module_t* backend_funcs_module)
{
    import_backend_methods(mod.julia_module(), backend_funcs_module, {
        "wrap_stream", "device_id", "stream_ptr", "memory_info"
    });

    mod.method("wrap_stream", [](void* cuda_stream){ return Kokkos::Cuda((cudaStream_t) cuda_stream); });
#if KOKKOS_VERSION_CMP(>=, 4, 0, 0)
    mod.method("device_id", [](){ return Kokkos::Impl::CudaInternal::m_cudaDev; });
#else
    mod.method("device_id", [](){ return Kokkos::Impl::CudaInternal::singleton().m_cudaDev; });
#endif
    mod.method("device_id", [](const Kokkos::Cuda& s){ return s.cuda_device(); });
    mod.method("stream_ptr", [](const Kokkos::Cuda& s){ return (void*) s.cuda_stream(); });

    mod.method("memory_info", [](){
        size_t free, total;
        CUresult res = cuMemGetInfo_v2(&free, &total);
        if (res != CUresult::CUDA_SUCCESS) {
            const char* error_msg;
            cuGetErrorString(res, &error_msg);
            if (error_msg == nullptr) {
                error_msg = "<could not get error message>";
            }
            jl_errorf("CUDA error when calling `cuMemGetInfo_v2`: %s", error_msg);
        }
        return std::make_tuple(free, total);
    });
}
#endif


#ifdef KOKKOS_ENABLE_HIP
template<>
void space_methods<SpaceInfo<Kokkos_HIP::HIP>>(jlcxx::Module& mod, jl_module_t* backend_funcs_module)
{
    import_backend_methods(mod.julia_module(), backend_funcs_module, {
        "wrap_stream", "device_id", "stream_ptr", "memory_info"
    });

    mod.method("wrap_stream", [](void* hip_stream){ return Kokkos_HIP::HIP((hipStream_t) hip_stream); });
#if KOKKOS_VERSION_CMP(>=, 4, 0, 0)
    mod.method("device_id", [](){ return Kokkos_HIP::Impl::HIPInternal::m_hipDev; });
#else
    mod.method("device_id", [](){ return Kokkos_HIP::Impl::HIPInternal::singleton().m_hipDev; });
#endif
    mod.method("device_id", [](const Kokkos_HIP::HIP& s){ return s.hip_device(); });
    mod.method("stream_ptr", [](const Kokkos_HIP::HIP& s){ return (void*) s.hip_stream(); });

    mod.method("memory_info", [](){
        size_t free, total;
        hipError_t res = hipMemGetInfo(&free, &total);
        if (res != hipSuccess) {
            const char* error_msg = hipGetErrorString(res);
            if (error_msg == nullptr) {
                error_msg = "<could not get error message>";
            }
            jl_errorf("HIP error when calling `hipMemGetInfo`: %s", error_msg);
        }
        return std::make_tuple(free, total);
    });
}
#endif


template<typename... T>
void register_all(jlcxx::Module& mod, jl_module_t* spaces_module, TList<T...>)
{
    (space_methods<SpaceInfo<T>>(mod, spaces_module), ...);
}


void define_space_specific_methods(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* kokkos_module = wrapper_module->parent;
    auto* backend_funcs_module = (jl_module_t*) jl_get_global(kokkos_module, jl_symbol("BackendFunctions"));

    mod.set_override_module(backend_funcs_module);
    register_all(mod, backend_funcs_module, ExecutionSpaceList{});
    mod.unset_override_module();
}
