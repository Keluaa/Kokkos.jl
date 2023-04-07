
#include "views.h"
#include "memory_spaces.h"

#include <type_traits>


using jlcxx::Parametric;
using jlcxx::TypeVar;

using DimensionsToInstantiate = std::integer_sequence<int, VIEW_DIMENSIONS>;

const size_t KOKKOS_MAX_DIMENSIONS = 8;


/**
 * Sets `type` to `T` followed by `N` pointers: `add_pointers<int, 2>::type == int**`
 */
template<typename T, int N>
struct add_pointers { using type = typename add_pointers<std::add_pointer_t<T>, N-1>::type; };

template<typename T>
struct add_pointers<T, 0> { using type = T; };


/**
 * Basic wrapper around a `Kokkos::View`, mostly providing convenience functionalities over dimensions and the data type
 * of the view.
 */
template<typename T, typename DimCst, typename MemSpace, typename T_Ptr = typename add_pointers<T, DimCst::value>::type>
struct ViewWrap : public Kokkos::View<T_Ptr, MemSpace>
{
    using type = T;
    using mem_space = MemSpace;
    using Kokkos::View<T_Ptr, MemSpace>::View;

    static constexpr size_t dim = DimCst::value;

    using IdxTuple [[maybe_unused]] = decltype(std::tuple_cat(std::array<Idx, dim>()));

    [[nodiscard]] std::array<int64_t, dim> get_dims() const {
        std::array<int64_t, dim> dims{};
        for (size_t i = 0; i < dim; i++) {
            dims.at(i) = this->extent_int(i);
        }
        return dims;
    }
};


bool is_integer(jl_datatype_t* v)
{
    return v == jl_int64_type || v == jl_int32_type || v == jl_uint64_type || v == jl_uint32_type;
}


size_t unbox_dim_size(jl_datatype_t* type, const uint64_t* v)
{
    constexpr size_t LOW_32BITS_MASK = 0xFF'FF'FF'FF;
    if (type == jl_int64_type) return *v;
    if (type == jl_int32_type) return *v & LOW_32BITS_MASK;
    if (type == jl_uint64_type) return *v;
    if (type == jl_uint32_type) return *v & LOW_32BITS_MASK;
    jl_error("unknown dimension integer type argument");
}


std::array<size_t, KOKKOS_MAX_DIMENSIONS> unpack_dims(size_t dim, jl_value_t* dims)
{
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

    auto* dims_t = (jl_datatype_t*) jl_typeof(dims);

    if (is_integer(dims_t)) {
        N.at(0) = jl_unbox_uint64(dims);
    } else if (jl_is_tuple(dims)) {
        size_t const len = jl_nparams(dims_t);
        if (len > dim) {
            jl_errorf("too many dimensions for Kokkos.View{T, %d} constructor: %d (maximum is %d)", dim, len, dim);
        }
        for (size_t i = 0; i < len; i++) {
            auto* field_ptr = (uint64_t*) (((char*) jl_data_ptr(dims)) + jl_field_offset(dims_t, (int) i));
            auto* d_t = (jl_datatype_t*) jl_tparam(dims_t, i);
            if (is_integer(d_t)) {
                N.at(i) = unbox_dim_size(d_t, field_ptr);
            } else {
                jl_errorf("dimension %d in Kokkos.View{T, %d} constructor doesn't have a valid integer type", i+1, dim);
            }
        }
    } else {
        jl_errorf("incompatible type for `dims` argument of Kokkos.View{T, %d} constructor: %s",
                  dim, jl_typename_str((jl_value_t*) dims_t));
    }

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
        jl_type_error_rt("Kokkos.View", "memory space",  (jl_value_t*) jlcxx::julia_type<MemSpace>(), boxed_memory_space);
    }
}


/**
 * Returns a `Kokkos::View<add_pointers<T, Dim>, MemSpace>>` with the given `label` and dimensions.
 * `init` specifies whether or not to zero-fill the view at initialization. Very important for first-touch optimizations.
 * `pad` specifies whether or not to allow padding of dimensions.
 *
 * `dims` is a Julia value:
 *  - a boxed integer (Int32, Int64, UInt32 or UInt64): dimension of the first dimension, only for 1D views
 *  - a tuple of integers (Int32, Int64, UInt32 or UInt64): dimensions of each dimensions
 *
 * `boxed_memory_space` is a Julia value:
 *  - `nothing`: default construct the memory space object in-place
 *  - an instance of `MemorySpace`: use the boxed C++ instance
 */
template<typename T, typename DimCst, typename MemSpace>
ViewWrap<T, DimCst, MemSpace> create_view(jl_value_t* dims, jl_value_t* boxed_memory_space, const char* label, bool init, bool pad)
{
    static_assert(DimCst::value <= KOKKOS_MAX_DIMENSIONS, "Kokkos supports only up to 8 dimensions");

    auto [N0, N1, N2, N3, N4, N5, N6, N7] = unpack_dims(DimCst::value, dims);
    const std::string label_str(label);

    const auto* mem_space_p = unbox_memory_space_arg<MemSpace>(boxed_memory_space);
    const auto& mem_space = (mem_space_p == nullptr) ? MemSpace{} : *mem_space_p;

    if (init) {
        if (pad) {
            auto ctor_prop = Kokkos::view_alloc(label_str, mem_space, Kokkos::AllowPadding);
            return ViewWrap<T, DimCst, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
        } else {
            auto ctor_prop = Kokkos::view_alloc(label_str, mem_space);
            return ViewWrap<T, DimCst, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
        }
    } else {
        if (pad) {
            auto ctor_prop = Kokkos::view_alloc(label_str, mem_space, Kokkos::WithoutInitializing, Kokkos::AllowPadding);
            return ViewWrap<T, DimCst, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
        } else {
            auto ctor_prop = Kokkos::view_alloc(label_str, mem_space, Kokkos::WithoutInitializing);
            return ViewWrap<T, DimCst, MemSpace>(ctor_prop, N0, N1, N2, N3, N4, N5, N6, N7);
        }
    }
}


jl_value_t* build_abstract_array_type(jlcxx::Module& mod, int dim, jl_datatype_t* space_type)
{
    jl_module_t* kokkos_module = mod.julia_module();

    // Since we call `mod.add_type` by applying only the data type of the array, we need a UnionAll with the dimension
    // already specified. The Julia equivalent would be `Kokkos.View{T, dim, space_type} where T`.

    jl_value_t** stack;
    JL_GC_PUSHARGS(stack, 5);

    // `T_var = TypeVar(:T)`
    jl_tvar_t* T_var = jl_new_typevar(jl_symbol("T"), jl_bottom_type, (jl_value_t*) jl_any_type);
    stack[0] = (jl_value_t*) T_var;

    jl_value_t* boxed_dim = jl_box_int64(dim);
    stack[1] = boxed_dim;

    stack[2] = (jl_value_t*) space_type;

    // `Kokkos.View`
    jl_value_t* view_t = jl_get_global(kokkos_module, jl_symbol("View"));
    stack[3] = view_t;

    // `Kokkos.View{T_var, dim, space_type}`
    jl_value_t* view_data_type = jl_apply_type(view_t, stack, 3);
    stack[4] = view_data_type;

    // `Kokkos.View{T_var, dim, space_type} where T_var`
    jl_value_t* view_union_all = jl_type_unionall(T_var, view_data_type);

    JL_GC_POP();

    return view_union_all;
}


struct RegisterUtils
{
    template<std::size_t>
    using inner_type = Idx;

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

    template<size_t D, typename MemorySpace, typename Wrapped>
    static void register_access_operator(Wrapped wrapped) {
        if constexpr (Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, MemorySpace>::accessible) {
            register_access_operator(wrapped, std::make_index_sequence<D>{});
        } else {
            register_inaccessible_operator(wrapped, std::make_index_sequence<D>{});
        }
    }
};


template<typename MemorySpace, size_t D>
void register_view_types(jlcxx::Module& mod)
{
    using view_dim = std::integral_constant<size_t, D>;

    jl_value_t* view_type = build_abstract_array_type(mod, D, jlcxx::julia_type<MemorySpace>());

    std::stringstream str;
    str << "View" << D << "D_" << MemorySpace::name();
    auto name = str.str();

    // We apply the type and dimension separately: some type problems arise when specifying both through `add_type`,
    // irregularities like `View{Float64, 2} <: AbstractArray{Float64, 2} == true` but an instance of a
    // `View{Float64, 2}` would not be `isa AbstractArray{Float64, 2}`, preventing the inheritance of all AbstractArray
    // behaviour.
    mod.add_type<Parametric<TypeVar<1>>>(name, view_type)
            .apply_combination<
                    ViewWrap, jlcxx::ParameterList<VIEW_TYPES>, jlcxx::ParameterList<view_dim>, jlcxx::ParameterList<MemorySpace>
            >([&](auto wrapped)
    {
        using WrappedT = typename decltype(wrapped)::type;
        using type = typename WrappedT::type;

        wrapped.method("alloc_view",
        [](jlcxx::SingletonType<type>, jlcxx::Val<int64_t, D>, jlcxx::SingletonType<SpaceInfo<MemorySpace>>,
                jl_value_t* dims, jl_value_t* boxed_memory_space, const char* label, bool init, bool pad)
        {
            return create_view<type, view_dim, MemorySpace>(dims, boxed_memory_space, label, init, pad);
        });
        wrapped.method("view_size", &WrappedT::size);
        wrapped.method("view_data", &WrappedT::data);
        wrapped.method("label", &WrappedT::label);
        RegisterUtils::register_access_operator<D, MemorySpace>(wrapped);
        wrapped.method("dimension", [](const WrappedT&) { return D; });
        wrapped.method("get_dims", [](const WrappedT& view)
        {
            auto dims = view.get_dims();
            return std::tuple_cat(dims);
        });
    });
}


template<typename MemorySpace, typename T, T... I>
void register_all_dimensions(jlcxx::Module& mod, std::integer_sequence<T, I...>)
{
    ([&](){ register_view_types<MemorySpace, I>(mod); } (), ...);
}


template<typename... T>
void register_views_for_all_memory_spaces(jlcxx::Module& mod, MemorySpaces<T...>)
{
    ([&](){ register_all_dimensions<T>(mod, DimensionsToInstantiate{}); } (), ...);
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


JLCXX_MODULE define_kokkos_views(jlcxx::Module& mod)
{
    mod.method("idx_type", &get_idx_type);
    mod.method("compiled_dims", [](){ return std::make_tuple(VIEW_DIMENSIONS); });
    mod.method("compiled_types", [](){ return std::tuple_cat(build_julia_types_array()); });
    register_views_for_all_memory_spaces(mod, MemorySpacesList{});
}
