
#include "kokkos_wrapper.h"

#include "spaces.h"
#include "layouts.h"

#include <sstream>


void kokkos_init(const Kokkos::InitializationSettings& settings)
{
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize(settings);
    } else {
        jl_error("Kokkos is already initialized");
    }
}


void kokkos_finalize()
{
    if (!Kokkos::is_initialized()) {
        jl_error("Kokkos is not initialized");
    } else if (Kokkos::is_finalized()) {
        jl_error("Kokkos is already finalized initialized");
    } else {
        Kokkos::finalize();
    }
}


jl_value_t* kokkos_version()
{
    auto* version_number_t = (jl_function_t*) jl_get_global(jl_base_module, jl_symbol("VersionNumber"));

    jl_value_t** versions;
    JL_GC_PUSHARGS(versions, 3);
#ifndef KOKKOS_VERSION_MAJOR
    versions[0] = jl_box_int64(KOKKOS_VERSION / 10000);
    versions[1] = jl_box_int64(KOKKOS_VERSION / 100 % 100);
    versions[2] = jl_box_int64(KOKKOS_VERSION % 100);
#else
    versions[0] = jl_box_int64(KOKKOS_VERSION_MAJOR);
    versions[1] = jl_box_int64(KOKKOS_VERSION_MINOR);
    versions[2] = jl_box_int64(KOKKOS_VERSION_PATCH);
#endif
    jl_value_t* version = jl_call(version_number_t, versions, 3);
    JL_GC_POP();

    return version;
}


void define_initialization_settings(jlcxx::Module& mod)
{
    auto settings_t = mod.add_type<Kokkos::InitializationSettings>("InitializationSettings")
        .constructor<>()
        .method("num_threads!", &Kokkos::InitializationSettings::set_num_threads)
        .method("device_id!", &Kokkos::InitializationSettings::set_device_id)
        .method("disable_warnings!", &Kokkos::InitializationSettings::set_disable_warnings)
        .method("print_configuration!", &Kokkos::InitializationSettings::set_print_configuration)
        .method("tune_internals!", &Kokkos::InitializationSettings::set_tune_internals)
        .method("tools_libs!", &Kokkos::InitializationSettings::set_tools_libs)
        .method("tools_args!", &Kokkos::InitializationSettings::set_tools_args)
        .method("map_device_id_by!", [](Kokkos::InitializationSettings& settings, jl_value_t* val)
        {
            if (!jl_is_symbol(val)) {
                jl_type_error("map_device_id_by!", (jl_value_t*) jl_symbol_type, val);
            }
            auto* sym = (jl_sym_t*) val;
            if (sym == jl_symbol("mpi_rank")) {
                settings.set_map_device_id_by("mpi_rank");
            } else if (sym == jl_symbol("random")) {
                settings.set_map_device_id_by("random");
            } else {
                jl_errorf("expected `:mpi_rank` or `:random`, got: `:%s`", jl_symbol_name(sym));
            }
            return settings;
        });

    // Getters must account for std::optional. We return `nothing` when the value is not set
#define SETTINGS_GETTER(name)                                                                        \
        settings_t.method(#name, [](const Kokkos::InitializationSettings& settings)                  \
        {                                                                                            \
            if (settings.has_##name())                                                               \
                return jlcxx::box<decltype(settings.get_##name())>(settings.get_##name());           \
            return jl_nothing;                                                                       \
        })

    SETTINGS_GETTER(device_id);
    SETTINGS_GETTER(num_threads);
    SETTINGS_GETTER(disable_warnings);
    SETTINGS_GETTER(print_configuration);
    SETTINGS_GETTER(tune_internals);
#undef SETTINGS_GETTER

    settings_t.method("map_device_id_by", [](const Kokkos::InitializationSettings& settings) {
        if (settings.has_map_device_id_by())
            return (jl_value_t*) jl_symbol(settings.get_map_device_id_by().c_str());
        return jl_nothing;
    });

    settings_t.method("tools_libs", [](const Kokkos::InitializationSettings& settings) {
        if (settings.has_tools_libs())
            return jl_cstr_to_string(settings.get_tools_libs().c_str());
        return jl_nothing;
    });

    settings_t.method("tools_args", [](const Kokkos::InitializationSettings& settings) {
        if (settings.has_tools_args())
            return jl_cstr_to_string(settings.get_tools_args().c_str());
        return jl_nothing;
    });
}


void print_configuration(jl_value_t* io, bool verbose)
{
    jl_value_t* IO_t = jl_get_global(jl_core_module, jl_symbol("IO"));
    if (!jl_isa(io, IO_t))
        jl_type_error("print_configuration", IO_t, io);

    std::ostringstream oss;
    Kokkos::print_configuration(oss, verbose);
    const std::string config_str = oss.str();

    jl_function_t* println = jl_get_function(jl_base_module, "println");
    jl_call2(println, io, jl_cstr_to_string(config_str.c_str()));
}


void import_all_env_methods(jl_module_t* impl_module, jl_module_t* kokkos_module)
{
    // In order to override the methods in the main Kokkos module, we must have them imported
    const std::array declared_methods = {
            "print_configuration",
            "initialize",
            "finalize",
            "fence",
            "num_threads",
            "device_id",
            "disable_warnings",
            "print_configuration",
            "tune_internals",
            "tools_libs",
            "tools_args",
            "map_device_id_by",
            "num_threads!",
            "device_id!",
            "disable_warnings!",
            "print_configuration!",
            "tune_internals!",
            "tools_libs!",
            "tools_args!",
            "map_device_id_by!"
    };

    for (auto& method : declared_methods) {
        jl_module_import(impl_module, kokkos_module, jl_symbol(method));
    }
}


void register_view_finalizer(jl_module_t* kokkos_module)
{
    auto views_module = (jl_module_t*) jl_get_global(kokkos_module, jl_symbol("Views"));
    jl_value_t* finalize_all_views = jl_get_global(views_module, jl_symbol("_finalize_all_views"));
    if (finalize_all_views == nullptr) {
        jl_error("could not get `Kokkos.Views._finalize_all_views`");
    }

    Kokkos::push_finalize_hook([=](){
        bool was_adopted = false;
        if (jl_get_pgcstack() == nullptr) {
            // Some Kokkos app called `Kokkos::finalize` from another thread.
            jl_adopt_thread();
            was_adopted = true;
        }

        jl_call0(finalize_all_views);
        if (jl_exception_occurred()) {
            fprintf(stderr, "Error in Kokkos::finalize hook for `Kokkos.jl`, in `Kokkos.Views._finalize_all_views`: %s.\n"
                            "All views might not be freed correctly.",
                    jl_typeof_str(jl_current_exception()));
        }

        if (was_adopted) {
            // Mark the thread as GC safe until the end of time
            jl_gc_safe_enter(jl_current_task->ptls);
        }
    });
}


JLCXX_MODULE define_kokkos_module(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    jl_module_t* kokkos_module = wrapper_module->parent;

    import_all_env_methods(mod.julia_module(), kokkos_module);

    mod.set_override_module(kokkos_module);

    define_initialization_settings(mod);
    mod.method("print_configuration", &print_configuration);

    mod.method("initialize", &kokkos_init);
    mod.method("finalize", &kokkos_finalize);

    mod.method("is_initialized",  (bool (*)()) &Kokkos::is_initialized);
    mod.method("is_finalized", (bool (*)()) &Kokkos::is_finalized);

    mod.method("fence", [](){ Kokkos::fence(); });
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
    mod.method("fence", [](const std::string& s){ Kokkos::fence(s); });
#else
    mod.method("fence", &Kokkos::fence);
#endif // __INTEL_COMPILER

    mod.unset_override_module();

    mod.method("__kokkos_version", &kokkos_version);

    register_view_finalizer(kokkos_module);

    define_all_layouts(mod);
    define_all_spaces(mod);
}
