
#include "spaces.h"
#include "execution_spaces.h"
#include "memory_spaces.h"
#include "utils.h"
#include "layouts.h"


template<typename Space>
void register_space(jlcxx::Module& mod, jl_module_t* spaces_module)
{
    const std::string main_type_name = std::string(SpaceInfo<Space>::julia_name);
    const std::string impl_type_name = main_type_name + "Impl";

    // Manual 'mod.map_type' for 'SpaceInfo<Space>' since the type is in another module
    jl_value_t* main_type_dt = jlcxx::julia_type(main_type_name, spaces_module);
    if (main_type_dt == nullptr) {
        throw std::runtime_error("Type '" + main_type_name + "' not found in the Kokkos module");
    }
    jlcxx::set_julia_type<SpaceInfo<Space>>((jl_datatype_t *) main_type_dt);

    // In Kokkos 4.2.00, `Kokkos::HostSpace` became a trivial type, therefore the default `JlCxx::IsMirroredType` became
    // `true`, meaning that `map_type` should be used instead of `add_type`. For uniformity (GPU spaces will always be
    // non-trivial types because of streams), all spaces are enforced to be non-mirrored types, they are all declared as
    // `jlCxx::IsMirroredType : std::false_type`.
    auto space_type = mod.add_type<Space>(impl_type_name, jlcxx::julia_type<SpaceInfo<Space>>());
    space_type.constructor();

    if constexpr (Kokkos::is_memory_space<Space>::value) {
        space_type.method("allocate", [](const Space& s, ptrdiff_t size) { return s.allocate(size); });
        space_type.method("deallocate", [](const Space& s, void* ptr, ptrdiff_t size) { return s.deallocate(ptr, size); });
    } else if constexpr (Kokkos::is_execution_space<Space>::value) {
        space_type.method("concurrency", [](const Space& s){ return s.concurrency(); });  // Serial::concurrency is static, while OpenMP::concurrency is not
        space_type.method("fence", &Space::fence);
    }

    mod.method("kokkos_name", [](jlcxx::SingletonType<SpaceInfo<Space>>) { return std::string(Space::name()); });
    mod.method("enabled", [](jlcxx::SingletonType<SpaceInfo<Space>>) { return true; });
    mod.method("impl_space_type", [](jlcxx::SingletonType<SpaceInfo<Space>>) { return jlcxx::julia_type<Space>()->super; });
}


template<typename Space>
void post_register_space(jlcxx::Module& mod)
{
    // Important: instead of returning `jlcxx::julia_type<Space>` directly, we return its super-super-type (the main
    // type, defined in Julia independently of Kokkos flags) : this way we make sure that the original 'main types'
    // defined on the Julia side are returned, and not the one of the types defined internally by JlCxx.
    if constexpr (Kokkos::is_memory_space<Space>::value) {
        mod.method("execution_space", [](jlcxx::SingletonType<SpaceInfo<Space>>) {
            return jlcxx::julia_type<typename Space::execution_space>()->super->super;
        });
    } else if constexpr (Kokkos::is_execution_space<Space>::value) {
        mod.method("memory_space", [](jlcxx::SingletonType<SpaceInfo<Space>>) {
            return jlcxx::julia_type<typename Space::memory_space>()->super->super;
        });
        mod.method("array_layout", [](jlcxx::SingletonType<SpaceInfo<Space>>) {
            return jlcxx::julia_type<typename Space::array_layout>();
        });
    } else {
        static_assert(std::is_void_v<Space>, "'Space' is not an execution or memory space");
    }
}


template<typename T>
jl_datatype_t* get_julia_main_type(jl_module_t* spaces_module)
{
    return (jl_datatype_t*) jl_get_global(spaces_module, jl_symbol(SpaceInfo<T>::julia_name.data()));
}


template<typename... T>
void register_all(jlcxx::Module& mod, jl_module_t* spaces_module, TList<T...>, const std::string& spaces_name)
{
    (register_space<T>(mod, spaces_module), ...);

    mod.unset_override_module();
    const auto spaces_tuple = std::make_tuple(get_julia_main_type<T>(spaces_module)...);
    mod.method("__compiled_" + spaces_name, [=](){ return spaces_tuple; });
    mod.set_override_module(spaces_module);
}


template<typename... T>
void post_register_all(jlcxx::Module& mod, TList<T...>)
{
    (post_register_space<T>(mod), ...);

    // Accessibility from any space to all memory spaces
    apply_to_each(TList<T...>{}, [&](auto t_space) {
        using space = typename decltype(t_space)::template Arg<0>;
        apply_to_each(MemorySpacesList{}, [&](auto t_mem_space) {
            using mem_space = typename decltype(t_mem_space)::template Arg<0>;
            mod.method("accessible",
            [](jlcxx::SingletonType<SpaceInfo<space>>, jlcxx::SingletonType<SpaceInfo<mem_space>>) {
                return (bool) Kokkos::SpaceAccessibility<space, mem_space>::accessible;
            });
        });
    });
}


void define_execution_spaces_functions(jlcxx::Module& mod)
{
    mod.method("__default_device_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultExecutionSpace>()->super->super;
    });
    mod.method("__default_host_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultHostExecutionSpace>()->super->super;
    });
    mod.method("__idx_type", [](){
        return jlcxx::julia_base_type<Idx>();
    });
}


void define_memory_spaces_functions(jlcxx::Module& mod)
{
    mod.method("__default_memory_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultExecutionSpace::memory_space>()->super->super;
    });
    mod.method("__default_host_memory_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultHostExecutionSpace::memory_space>()->super->super;
    });
    mod.method("__shared_memory_space", [](){
#if KOKKOS_VERSION_CMP(>=, 4, 0, 0)
        if constexpr (Kokkos::has_shared_space) {
            return jlcxx::julia_type<Kokkos::SharedSpace>()->super->super;
        } else
#endif // KOKKOS_VERSION_CMP(>=, 4, 0, 0)
        {
            return jl_nothing;
        }
    });
    mod.method("__shared_host_pinned_space", [](){
#if KOKKOS_VERSION_CMP(>=, 4, 0, 0)
        if constexpr (Kokkos::has_shared_host_pinned_space) {
            return jlcxx::julia_type<Kokkos::SharedHostPinnedSpace>()->super->super;
        } else
#endif // KOKKOS_VERSION_CMP(>=, 4, 0, 0)
        {
            return jl_nothing;
        }
    });
}


void import_all_spaces_methods(jl_module_t* impl_module, jl_module_t* spaces_module)
{
    // In order to override the methods in the Kokkos module, we must have them imported
    const std::array declared_methods = {
        "allocate",
        "deallocate",
        "concurrency",
        "fence",
        "kokkos_name",
        "enabled",
        "impl_space_type",
        "execution_space",
        "accessible",
        "memory_space",
        "array_layout"
    };

    for (auto& method : declared_methods) {
        jl_module_import(impl_module, spaces_module, jl_symbol(method));
    }
}


void define_all_spaces(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* kokkos_module = wrapper_module->parent;

    import_all_spaces_methods(mod.julia_module(), kokkos_module);

    mod.set_override_module(kokkos_module);

    register_all(mod, kokkos_module, MemorySpacesList{}, "mem_spaces");
    register_all(mod, kokkos_module, ExecutionSpaceList{}, "exec_spaces");

    // Those functions need all execution and memory spaces to be registered
    post_register_all(mod, MemorySpacesList{});
    post_register_all(mod, ExecutionSpaceList{});

    mod.unset_override_module();

    define_execution_spaces_functions(mod);
    define_memory_spaces_functions(mod);

    define_space_specific_methods(mod);
}
