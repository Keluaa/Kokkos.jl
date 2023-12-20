
#include "utils.h"
#include "memory_spaces.h"
#include "layouts.h"
#include "views.h"

#include <variant>


using SubViewDimension = std::integral_constant<int, SUBVIEW_DIM>;


struct Colon_t {};  // Mapped to 'Base.Colon'
struct AbstractUnitRange_t {};  // Mapped to 'Base.AbstractUnitRange{Int64}'
struct IndexVarargs {};  // Mapped to 'Tuple{Vararg{Union{Base.Colon, Base.AbstractUnitRange{Int64}, Int64}}}'


// Range type used by Kokkos for Kokkos::ALL. Maybe a 'Kokkos::pair' instead, idk.
// See https://kokkos.github.io/kokkos-core-wiki/API/core/view/subview.html
using Range = std::pair<ptrdiff_t, ptrdiff_t>;


void setup_type_mappings()
{
    if (!jlcxx::has_julia_type<IndexVarargs*>()) {
        auto idx_t = (jl_datatype_t*) jl_eval_string("Tuple{Vararg{Union{Colon, AbstractUnitRange{Int64}, Int64}}}");
        if (idx_t == nullptr)
            jl_rethrow();
        jlcxx::set_julia_type<IndexVarargs*>(idx_t);
    }

    if (!jlcxx::has_julia_type<Colon_t>()) {
        auto colon_t = (jl_datatype_t*) jl_get_global(jl_base_module, jl_symbol("Colon"));
        jlcxx::set_julia_type<Colon_t>(colon_t);
    }

    if (!jlcxx::has_julia_type<AbstractUnitRange_t>()) {
        jl_value_t* range_union_all = jl_get_global(jl_base_module, jl_symbol("AbstractUnitRange"));
        jl_value_t* range_t = jl_apply_type1(range_union_all, (jl_value_t*) jlcxx::julia_base_type<int64_t>());
        jlcxx::set_julia_type<AbstractUnitRange_t>((jl_datatype_t*) range_t);
    }
}


template<size_t Dim, typename View>
size_t jl_indexes_to_cpp(jl_value_t* jl_indexes,
                         std::array<std::variant<int64_t, Range>, Dim>& indexes,
                         const View& view)
{
    // 'jl_indexes' is of type 'Tuple{Vararg{Union{Colon, AbstractUnitRange{Int64}, Int64}}}'
    jl_value_t* indexes_type = jl_typeof(jl_indexes);
    int index_count = jl_nparams(indexes_type);
    if (static_cast<size_t>(index_count) > Dim) {
        jl_errorf("expected a tuple of %d indexes or slices, got %d", Dim, index_count);
    }

    int int_count = 0;
    int r = -1;
    for (auto& index : indexes) {
        r++;

        if (r >= index_count) {
            // Select the whole dimension by default (Kokkos::ALL)
            index = Range{0, view.extent(r)};
            continue;
        }

        // ':'   => Range(0,v.extent(r))  (equiv to Kokkos::ALL)
        // 'a:b' => Range(a-1, b)
        // 'r'   => r-1

        auto* idx_type = (jl_datatype_t*) jl_tparam(indexes_type, r);

        if (idx_type == jlcxx::julia_base_type<int64_t>()) {
            int64_t val = jl_unbox_int64(jl_get_nth_field(jl_indexes, r));
            val -= 1;
            index = val;
            int_count++;
            if (!(0 <= val && static_cast<size_t>(val) < view.extent(r))) {
                jl_bounds_error(jlcxx::box<View>(view), jl_indexes);
            }
        } else if (idx_type == jlcxx::julia_type<Colon_t>()) {
            index = Range{0, view.extent(r)}; // Kokkos::ALL
        } else if (jl_subtype((jl_value_t*) idx_type, (jl_value_t*) jlcxx::julia_type<AbstractUnitRange_t>())) {
            jl_value_t* jl_range = jl_get_nth_field(jl_indexes, r);

            jl_value_t* jl_start = jl_call1(jl_get_global(jl_base_module, jl_symbol("first")), jl_range);
            int64_t start = jl_unbox_int64(jl_start);
            start -= 1;

            jl_value_t* jl_end = jl_call1(jl_get_global(jl_base_module, jl_symbol("last")), jl_range);
            int64_t end = jl_unbox_int64(jl_end);

            index = Range{start, end};
            if (!(0 <= start && start <= end && static_cast<size_t>(end) <= view.extent(r))) {
                jl_bounds_error(jlcxx::box<View>(view), jl_indexes);
            }
        } else {
            jl_errorf("Expected a value of type Union{Colon, AbstractUnitRange{Int64}, Int64}, got: %s",
                      jl_typename_str((jl_value_t*) idx_type));
        }
    }

    return int_count;
}


template<typename T, typename... V>
constexpr std::size_t count_same()
{
    return (std::is_same_v<T, V> + ...);
}


template<typename View, typename SubView, typename... Indexes>
SubView do_subview(const View& view, const std::tuple<Indexes...>& indexes_tuple)
{
    return std::apply([&](const Indexes&... indexes) {
        // 'std::visit' will automatically instantiate all combinations of the possible values of the variants.
        // Therefore, we must filter them by checking if the variant combination matches with the expected subview
        // dimension.
        return std::visit([&](auto&&... idx) {
            constexpr auto int_count = count_same<int64_t, std::decay_t<decltype(idx)>...>();
            if constexpr ((View::dim - int_count) == SubView::dim) {
                if constexpr (std::is_same_v<typename View::layout, typename SubView::layout> &&
                        !std::is_same_v<typename SubView::kokkos_view_t, Kokkos::Subview<View, std::decay_t<decltype(idx)>...>>) {
                    // Kokkos::subview would have returned an incompatible subview type
                    jl_errorf("Internal subview call error. Expected a Kokkos::Subview type of '%s', got '%s'",
                              typeid(typename SubView::kokkos_view_t).name(),
                              typeid(Kokkos::Subview<View, std::decay_t<decltype(idx)>...>).name());
                    // Dummy return value, an empty view with a defaulted layout. This code is unreachable.
                    return SubView("ERROR", typename SubView::layout{});
                } else {
                    return SubView(Kokkos::subview(view, idx...));
                }
            } else {
                jl_errorf("Internal subview call error. Expected %d integers in indexes list, got %d",
                          View::dim - SubView::dim, int_count);
                // Dummy return value, an empty view with a defaulted layout. This code is unreachable.
                return SubView("ERROR", typename SubView::layout{});
            }
        }, indexes...);
    }, indexes_tuple);
}


template<typename View, typename SubView, typename Layout>
void register_subviews_for_view_and_layout(jlcxx::Module& mod)
{
    if constexpr (!std::is_same_v<typename View::layout, Kokkos::LayoutStride>) {
        // A subview of a View with a LayoutLeft or LayoutRight can have a LayoutStride, which means that the return
        // value is different and therefore requires a separate method.
        using SubViewStrided = typename SubView::template with_layout<Kokkos::LayoutStride>;

        // method signature: (View{T, D, L, M}, Tuple{Vararg{Union{Colon, AbstractUnitRange, Int64}}}, Val{SubDim}, LayoutStride)
        mod.method("subview", [](const View& v, IndexVarargs* indexes,
                                 jlcxx::SingletonType<jlcxx::Val<int64_t, SubView::dim>>,
                                 jlcxx::SingletonType<Kokkos::LayoutStride>)
        {
            auto* jl_indexes = reinterpret_cast<jl_value_t*>(indexes);

            std::array<std::variant<int64_t, Range>, View::dim> view_indexes;
            size_t int_count = jl_indexes_to_cpp(jl_indexes, view_indexes, v);

            if (View::dim - int_count != SubViewStrided::dim) {
                jl_errorf("Expected %d integers in indexes list (to obtain a subview of dimension %d), got %d",
                          View::dim - SubViewStrided::dim, SubViewStrided::dim, int_count);
            }

            return do_subview<View, SubViewStrided>(v, std::tuple_cat(view_indexes));
        });
    }

    // method signature: (View{T, D, L, M}, Tuple{Vararg{Union{Colon, AbstractUnitRange, Int64}}}, Val{SubDim}, Layout)
    mod.method("subview", [](const View& v, IndexVarargs* indexes,
                             jlcxx::SingletonType<jlcxx::Val<int64_t, SubView::dim>>,
                             jlcxx::SingletonType<typename View::layout>)
    {
        auto* jl_indexes = reinterpret_cast<jl_value_t*>(indexes);

        std::array<std::variant<int64_t, Range>, View::dim> view_indexes;
        size_t int_count = jl_indexes_to_cpp(jl_indexes, view_indexes, v);

        if (View::dim - int_count != SubView::dim) {
            jl_errorf("Expected %d integers in indexes list (to obtain a subview of dimension %d), got %d",
                      View::dim - SubView::dim, SubView::dim, int_count);
        }

        return do_subview<View, SubView>(v, std::tuple_cat(view_indexes));
    });
}


void register_all_subviews(jlcxx::Module& mod)
{
    auto view_combinations = build_all_combinations<
            FilteredMemorySpaceList
    >();

    apply_to_all(view_combinations, [&](auto view_t) {
        using MemSpace = typename decltype(view_t)::template Arg<0>;

        using View = ViewWrap<VIEW_TYPE, Dimension, Layout, MemSpace>;
        using SubView = ViewWrap<VIEW_TYPE, SubViewDimension, Layout, MemSpace>;

        if constexpr (SubViewDimension::value > Dimension::value) {
            jl_errorf("Expected a subview dimension lower than %d, got: %d",
                      Dimension::value, SubViewDimension::value);
        }
        else {
            if (!jlcxx::has_julia_type<SubView>()) {
                jl_errorf("Missing view type for complete `Kokkos.subview` coverage: %dD of c++ type %s",
                          SubViewDimension::value, typeid(SubView).name());
            }

            register_subviews_for_view_and_layout<View, SubView, Layout>(mod);
        }
    });
}


JLCXX_MODULE define_kokkos_module(jlcxx::Module& mod)
{
    // Called from 'Kokkos.Views.Impl<number>'
    jl_module_t* views_module = mod.julia_module()->parent;
    jl_module_import(mod.julia_module(), views_module, jl_symbol("subview"));

    setup_type_mappings();
    register_all_subviews(mod);
    mod.method("params_string", get_params_string);
}
