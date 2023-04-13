
#include "spaces.h"
#include "execution_spaces.h"
#include "memory_spaces.h"


template<typename Space>
void register_space(jlcxx::Module& mod)
{
    const std::string main_type_name = SpaceInfo<Space>::julia_name;
    const std::string impl_type_name = main_type_name + "Impl";

    mod.map_type<SpaceInfo<Space>>(main_type_name);
    auto space_type = mod.add_type<Space>(impl_type_name, jlcxx::julia_type<SpaceInfo<Space>>());
    space_type.constructor();

    if constexpr (Kokkos::is_memory_space<Space>::value) {
        space_type.method("allocate", [](const Space& s, ptrdiff_t size) { return s.allocate(size); });
        space_type.method("deallocate", [](const Space& s, void* ptr, ptrdiff_t size) { return s.deallocate(ptr, size); });
    } else if constexpr (Kokkos::is_execution_space<Space>::value) {
        space_type.method("concurrency", &Space::concurrency);
        space_type.method("fence", &Space::fence);
    }

    mod.method("kokkos_name", [](jlcxx::SingletonType<SpaceInfo<Space>>) { return std::string(Space::name()); });
    mod.method("enabled", [](jlcxx::SingletonType<SpaceInfo<Space>>) { return true; });
}


template<typename Space>
void post_register_space(jlcxx::Module& mod)
{
    // Important: instead of returning `jlcxx::julia_type<Space>` directly, we return its super-super-type (the main
    // type, defined in Julia independently of Kokkos flags) : this way we make sure that the original 'main types'
    // defined on the Julia side are returned, and not the one of the types defined internally by JlCxx.
    if constexpr (Kokkos::is_memory_space<Space>::value) {
        mod.method("execution_space", [=](jlcxx::SingletonType<SpaceInfo<Space>>) {
            return jlcxx::julia_type<typename Space::execution_space>()->super->super;
        });
        mod.method("accessible", [](jlcxx::SingletonType<SpaceInfo<Space>>) {
            return (bool) Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, Space>::accessible;
        });
    } else if constexpr (Kokkos::is_execution_space<Space>::value) {
        mod.method("memory_space", [=](jlcxx::SingletonType<SpaceInfo<Space>>) {
            return jlcxx::julia_type<typename Space::memory_space>()->super->super;
        });
    } else {
        static_assert(std::is_void_v<Space>, "'Space' is not an execution or memory space");
    }
}


template<typename T>
jl_datatype_t* get_julia_main_type(jlcxx::Module& mod)
{
    return (jl_datatype_t*) jl_get_global(mod.julia_module(), jl_symbol(SpaceInfo<T>::julia_name));
}


template<template<typename> typename Container, typename... T>
void register_all(jlcxx::Module& mod, Container<T...>, const std::string& spaces_name)
{
    ([&](){ register_space<T>(mod); }(), ...);

    const auto spaces_tuple = std::make_tuple(get_julia_main_type<T>(mod)...);
    mod.method("compiled_" + spaces_name, [=](){ return spaces_tuple; });
}


template<template<typename> typename Container, typename... T>
void post_register_all(jlcxx::Module& mod, Container<T...>)
{
    ([&](){ post_register_space<T>(mod); }(), ...);
}


void define_execution_spaces_functions(jlcxx::Module& mod)
{
    mod.method("default_device_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultExecutionSpace>()->super->super;
    });
    mod.method("default_host_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultHostExecutionSpace>()->super->super;
    });
}


void define_memory_spaces_functions(jlcxx::Module& mod)
{
    mod.method("default_memory_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultExecutionSpace::memory_space>()->super->super;
    });
    mod.method("default_host_memory_space", [](){
        return jlcxx::julia_type<Kokkos::DefaultHostExecutionSpace::memory_space>()->super->super;
    });
    mod.method("shared_memory_space", [](){
#ifdef KOKKOS_VERSION_GREATER_EQUAL
#if KOKKOS_VERSION_GREATER_EQUAL(4, 0, 0)
        if constexpr (Kokkos::has_shared_space) {
            return jlcxx::julia_type<Kokkos::SharedSpace>()->super->super;
        } else
#endif //  KOKKOS_VERSION_GREATER_EQUAL(4, 0, 0)
#endif // KOKKOS_VERSION_GREATER_EQUAL
        {
            return jl_nothing;
        }
    });
    mod.method("shared_host_pinned_space", [](){
#ifdef KOKKOS_VERSION_GREATER_EQUAL
#if KOKKOS_VERSION_GREATER_EQUAL(4, 0, 0)
        if constexpr (Kokkos::has_shared_host_pinned_space) {
            return jlcxx::julia_type<Kokkos::SharedHostPinnedSpace>()->super->super;
        } else
#endif // KOKKOS_VERSION_GREATER_EQUAL(4, 0, 0)
#endif // KOKKOS_VERSION_GREATER_EQUAL
        {
            return jl_nothing;
        }
    });
}


JLCXX_MODULE define_all_spaces(jlcxx::Module& mod)
{
    register_all<MemorySpaces>(mod, MemorySpacesList{}, "mem_spaces");
    register_all<ExecutionSpaces>(mod, ExecutionSpaceList{}, "exec_spaces");

    // Those functions need all execution and memory spaces to be registered
    post_register_all<MemorySpaces>(mod, MemorySpacesList{});
    post_register_all<ExecutionSpaces>(mod, ExecutionSpaceList{});

    define_execution_spaces_functions(mod);
    define_memory_spaces_functions(mod);
}
