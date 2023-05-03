
#include "views.h"
#include "memory_spaces.h"
#include "utils.h"

#include <type_traits>


const size_t KOKKOS_MAX_DIMENSIONS = 8;


template<typename... Dims>
std::array<size_t, KOKKOS_MAX_DIMENSIONS> unpack_dims(const std::tuple<Dims...>& dims)
{
    static_assert(sizeof...(Dims) <= KOKKOS_MAX_DIMENSIONS, "Kokkos supports only up to 8 dimensions");

    std::array<size_t, KOKKOS_MAX_DIMENSIONS> N = {
            KOKKOS_IMPL_CTOR_DEFAULT_ARG,
            KOKKOS_IMPL_CTOR_DEFAULT_ARG,
            KOKKOS_IMPL_CTOR_DEFAULT_ARG,
            KOKKOS_IMPL_CTOR_DEFAULT_ARG,
            KOKKOS_IMPL_CTOR_DEFAULT_ARG,
            KOKKOS_IMPL_CTOR_DEFAULT_ARG,
            KOKKOS_IMPL_CTOR_DEFAULT_ARG,
            KOKKOS_IMPL_CTOR_DEFAULT_ARG
    };

    std::apply(
    [&](const Dims&... dim)
    {
        std::size_t n{0};
        ((N[n++] = dim), ...);
    }, dims);

    return N;
}


template<typename MemSpace>
const MemSpace* unbox_memory_space_arg(jl_value_t* boxed_memory_space)
{
    if (jl_is_nothing(boxed_memory_space)) {
        return nullptr;
    } else if (jl_typeis(boxed_memory_space, jlcxx::julia_type<MemSpace>())) {
        return jlcxx::unbox<MemSpace*>(boxed_memory_space);
    } else {
        jl_type_error_rt("Kokkos.View constructor", "memory space assignment",
                         (jl_value_t*) jlcxx::julia_type<MemSpace>(), boxed_memory_space);
    }
}


template<typename Dimension, typename Layout, typename MemSpace>
struct RegisterUtils
{
    static constexpr size_t D = Dimension::value;

    template<std::size_t>
    using inner_type = Idx;


    static std::string build_view_type_name()
    {
        std::stringstream str;
        str << "View" << D << "D_";
        if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
            str << "L_";
        } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutRight>) {
            str << "R_";
        } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutStride>) {
            str << "S_";
        } else {
            static_assert(std::is_same_v<Layout, void>, "Unknown layout type");
        }
        str << MemSpace::name();
        return str.str();
    }


    static jl_value_t* build_abstract_array_type(jl_module_t* views_module)
    {
        // Since we call `mod.add_type` by applying only the data type of the array, we need a UnionAll with the
        // dimension already specified. The Julia equivalent would be `Kokkos.View{T, D, Layout, MemSpace} where T`.

        jl_value_t** stack;
        JL_GC_PUSHARGS(stack, 6);

        // `T_var = TypeVar(:T)`
        jl_tvar_t* T_var = jl_new_typevar(jl_symbol("T"), jl_bottom_type, (jl_value_t*) jl_any_type);
        stack[0] = (jl_value_t*) T_var;

        stack[1] = jl_box_int64(D);
        stack[2] = (jl_value_t*) jlcxx::julia_type<Layout>();
        stack[3] = (jl_value_t*) jlcxx::julia_type<MemSpace>();

        // `Kokkos.View`
        jl_value_t* view_t = jl_get_global(views_module, jl_symbol("View"));
        stack[4] = view_t;
        if (view_t == nullptr) {
            throw std::runtime_error("Type 'View' not found in the Kokkos.Views module");
        }

        // `Kokkos.View{T_var, dim, layout_type, space_type}`
        jl_value_t* view_data_type = jl_apply_type(view_t, stack, 4);
        stack[5] = view_data_type;

        // `Kokkos.View{T_var, dim, layout_type, space_type} where T_var`
        jl_value_t* view_union_all = jl_type_unionall(T_var, view_data_type);

        JL_GC_POP();

        return view_union_all;
    }


    template<typename T>
    static jl_datatype_t* build_array_constructor_type(jl_module_t* views_module)
    {
        jl_value_t** stack;
        JL_GC_PUSHARGS(stack, 5);

        stack[0] = (jl_value_t*) jlcxx::julia_type<T>();
        stack[1] = jl_box_int64(D);
        stack[2] = (jl_value_t*) jlcxx::julia_type<Layout>();
        stack[3] = (jl_value_t*) jlcxx::julia_type<SpaceInfo<MemSpace>>();

        jl_value_t* view_t = jl_get_global(views_module, jl_symbol("View"));
        stack[4] = view_t;

        jl_value_t* array_ctor_t = jl_apply_type(view_t, stack, 4);

        JL_GC_POP();

        return (jl_datatype_t*) array_ctor_t;
    }


    /**
     * Returns a `Kokkos::View<add_pointers<T, Dim>, MemSpace>>` with the given `label` and dimensions.
     * `init` specifies whether or not to zero-fill the view at initialization. Very important for first-touch optimizations.
     * `pad` specifies whether or not to allow padding of dimensions.
     *
     * `dims` is a Julia value:
     *  - a boxed integer (Int64): dimension of the first dimension, only for 1D views
     *  - a tuple of integers (Int64): dimensions of each dimensions
     *
     * `boxed_memory_space` is a Julia value:
     *  - `nothing`: default construct the memory space object in-place
     *  - an instance of `MemorySpace`: use the boxed C++ instance
     */
    template<typename T, typename... Dims>
    static ViewWrap<T, Dimension, Layout, MemSpace> create_view(const std::tuple<Dims...>& dims,
                                                                jl_value_t* boxed_memory_space,
                                                                const char* label, bool init, bool pad)
    {
        static_assert(D == sizeof...(Dims));

        auto [N0, N1, N2, N3, N4, N5, N6, N7] = unpack_dims(dims);
        const std::string label_str(label);

        const auto* mem_space_p = unbox_memory_space_arg<MemSpace>(boxed_memory_space);
        const auto& mem_space = (mem_space_p == nullptr) ? MemSpace{} : *mem_space_p;

        // TODO: LayoutStride case

        if (init) {
            if (pad) {
                auto ctor_prop = Kokkos::view_alloc(label_str, mem_space, Kokkos::AllowPadding);
                return ViewWrap<T, Dimension, Layout, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
            } else {
                auto ctor_prop = Kokkos::view_alloc(label_str, mem_space);
                return ViewWrap<T, Dimension, Layout, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
            }
        } else {
            if (pad) {
                auto ctor_prop = Kokkos::view_alloc(label_str, mem_space, Kokkos::WithoutInitializing, Kokkos::AllowPadding);
                return ViewWrap<T, Dimension, Layout, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
            } else {
                auto ctor_prop = Kokkos::view_alloc(label_str, mem_space, Kokkos::WithoutInitializing);
                return ViewWrap<T, Dimension, Layout, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
            }
        }
    }


    template<typename T, typename... Dims>
    static ViewWrap<T, Dimension, Layout, MemSpace> view_wrap(const std::tuple<Dims...>& dims, T* data_ptr)
    {
        static_assert(D == sizeof...(Dims));
        auto [N0, N1, N2, N3, N4, N5, N6, N7] = unpack_dims(dims);
        auto ctor_prop = Kokkos::view_wrap(data_ptr);
        return ViewWrap<T, Dimension, Layout, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
    }


    template<typename View>
    static void throw_inaccessible_error(View& view)
    {
        const std::string str = view.label();
        if (str.empty()) {
            jl_errorf("the view is inaccessible from the default host execution space");
        } else {
            jl_errorf("the view '%s' is inaccessible from the default host execution space", str.c_str());
        }
    }


    template<typename WrappedT, typename... I>
    static void inaccessible_view(WrappedT& view, I...) {
        throw_inaccessible_error(view);
    }


    template<typename Wrapped, std::size_t... S>
    static void register_access_operator(Wrapped wrapped, std::index_sequence<S...>)
    {
        using WrappedT = typename decltype(wrapped)::type;
        // Add a method for integer indexing: `get_ptr(i::Idx)` in 1D, `get_ptr(i::Idx, j::Idx)` in 2D, etc
        wrapped.method("get_ptr", &WrappedT::template operator()<inner_type<S>...>);
        // Add a method for tuple indexing: `get_ptr(t::Tuple{Idx})` in 1D, `get_ptr(t::Tuple{Idx, Idx})` in 2D, etc
        wrapped.method("get_ptr", [](WrappedT& view, const typename WrappedT::IdxTuple* I)
        {
            return view(std::get<S>(*I)...);
        });
    }


    template<typename Wrapped, std::size_t... S>
    static void register_inaccessible_operator(Wrapped wrapped, std::index_sequence<S...>)
    {
        using WrappedT = typename decltype(wrapped)::type;

        wrapped.method("get_ptr", &inaccessible_view<WrappedT, inner_type<S>...>);
        wrapped.method("get_ptr", [](WrappedT& view, const typename WrappedT::IdxTuple*)
        {
            throw_inaccessible_error(view);
        });
    }


    template<typename Wrapped>
    static void register_access_operator(Wrapped wrapped) {
        if constexpr (Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, MemSpace>::accessible) {
            register_access_operator(wrapped, std::make_index_sequence<D>{});
        } else {
            register_inaccessible_operator(wrapped, std::make_index_sequence<D>{});
        }
    }


    template<typename Wrapped>
    static void register_constructor(jlcxx::Module& mod, jl_module_t* views_module) {
        using type = typename Wrapped::type;
        using ctor_type = TList<Wrapped>;

        jl_datatype_t* view_ctor_type = build_array_constructor_type<type>(views_module);
        jlcxx::set_julia_type<ctor_type>(view_ctor_type);

        using DimsTuple = decltype(std::tuple_cat(std::array<int64_t, D>()));

        mod.method("alloc_view",
        [](jlcxx::SingletonType<ctor_type>, const DimsTuple& dims, jl_value_t* boxed_memory_space,
                const char* label, bool init, bool pad)
        {
            return create_view<type>(dims, boxed_memory_space, label, init, pad);
        });

        mod.method("view_wrap",
        [](jlcxx::SingletonType<ctor_type>, const DimsTuple& dims, type* data_ptr)
        {
            return view_wrap<type>(dims, data_ptr);
        });
    }
};


void register_all_view_combinations(jlcxx::Module& mod, jl_module_t* views_module)
{
    using memorySpacesParameterList = decltype(to_parameter_list(MemorySpacesList{}));
    using dimensionsList = decltype(wrap_dims(DimensionsToInstantiate{}));
    using layoutParameterList = decltype(to_parameter_list(LayoutList{}));

    MyApplyTypes{}.apply_combination<
            TList,
            memorySpacesParameterList,
            layoutParameterList,
            dimensionsList
    >([&](auto params) {
        using MemSpace = typename decltype(params)::template Arg<0>;
        using Layout = typename decltype(params)::template Arg<1>;
        using Dimension = typename decltype(params)::template Arg<2>;

        using RegUtils = RegisterUtils<Dimension, Layout, MemSpace>;

        std::string name = RegUtils::build_view_type_name();
        jl_value_t* view_type = RegUtils::build_abstract_array_type(views_module);

        // We apply the type and dimension separately: some type problems arise when specifying both through `add_type`,
        // irregularities like `View{Float64, 2} <: AbstractArray{Float64, 2} == true` but an instance of a
        // `View{Float64, 2}` would not be `isa AbstractArray{Float64, 2}`, preventing the inheritance of all
        // AbstractArray behaviour.
        mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>(name, view_type)
            .apply_combination<
                    ViewWrap,
                    jlcxx::ParameterList<VIEW_TYPES>,
                    jlcxx::ParameterList<Dimension>,
                    jlcxx::ParameterList<Layout>,
                    jlcxx::ParameterList<MemSpace>
        >([&](auto wrapped) {
            using Wrapped_t = typename decltype(wrapped)::type;

            RegUtils::template register_constructor<Wrapped_t>(mod, views_module);
            RegUtils::register_access_operator(wrapped);

            wrapped.method("view_data", &Wrapped_t::data);
            wrapped.method("label", &Wrapped_t::label);
            wrapped.method("memory_span", [](const Wrapped_t& view) { return view.impl_map().memory_span(); });
            wrapped.method("get_dims", [](const Wrapped_t& view) { return std::tuple_cat(view.get_dims()); });
            wrapped.method("get_tracker", [](const Wrapped_t& view) {
                if (view.impl_track().has_record()) {
                    return reinterpret_cast<void*>(view.impl_track().template get_record<void>()->data());
                } else {
                    return (void*) nullptr;
                }
            });
        });
    });
}


constexpr int view_types_count = std::tuple_size_v<std::tuple<VIEW_TYPES>>;
using julia_types = std::array<jl_value_t*, view_types_count>;

template<std::size_t... I, typename... T>
void add_julia_types(julia_types& array, std::integer_sequence<std::size_t, I...>, std::tuple<T...>)
{
    ([&](){ array[I] = (jl_value_t*) jlcxx::julia_base_type<T>(); } (), ...);
}


julia_types build_julia_types_array()
{
    const std::tuple<VIEW_TYPES> view_types{};
    const std::make_index_sequence<view_types_count> indexes{};
    julia_types array{};
    add_julia_types(array, indexes, view_types);
    return array;
}


jl_datatype_t* get_idx_type()
{
    return jlcxx::julia_base_type<Idx>();
}


void import_all_views_methods(jl_module_t* impl_module, jl_module_t* views_module)
{
    // In order to override the methods in the Kokkos.Views module, we must have them imported
    const std::array declared_methods = {
        "get_ptr",
        "alloc_view",
        "view_wrap",
        "view_data",
        "memory_span",
        "label",
        "get_dims",
        "get_tracker"
    };

    for (auto& method : declared_methods) {
        jl_module_import(impl_module, views_module, jl_symbol(method));
    }
}


void define_kokkos_views(jlcxx::Module& mod)
{
    jl_module_t* wrapper_module = mod.julia_module()->parent;
    auto* views_module = (jl_module_t*) jl_get_global(wrapper_module->parent, jl_symbol("Views"));

    import_all_views_methods(mod.julia_module(), views_module);

    mod.set_override_module(views_module);
    register_all_view_combinations(mod, views_module);
    mod.unset_override_module();

    mod.method("__idx_type", &get_idx_type);
    mod.method("__compiled_dims", [](){ return std::make_tuple(VIEW_DIMENSIONS); });
    mod.method("__compiled_types", [](){ return std::tuple_cat(build_julia_types_array()); });
}
