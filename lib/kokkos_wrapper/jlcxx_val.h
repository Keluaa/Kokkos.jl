
#ifndef KOKKOS_WRAPPER_JLCXX_VAL_H
#define KOKKOS_WRAPPER_JLCXX_VAL_H

#include "jlcxx/jlcxx.hpp"

// TODO: PR to libcxxwrap-julia + `map_c_arg_type(::Type{Val{T}}) where {T} = Any` in CxxWrap.jl at line 602


namespace jlcxx
{
    template<typename T, T v>
    struct ValType
    {
    };

    template<typename T, T v>
    using Val = ValType<T, v>;

    template<const std::string_view& str>
    using ValSym = ValType<const std::string_view&, str>;

    template<typename T, T v>
    struct static_type_mapping<ValType<T, v>>
    {
        using type = jl_datatype_t*;
    };

    template<typename T, T v>
    struct julia_type_factory<ValType<T, v>>
    {
        static inline jl_datatype_t* julia_type()
        {
            return apply_type(::jlcxx::julia_type("Val", jl_base_module), (jl_datatype_t*) ::jlcxx::box<T>(v));
        }
    };

    template<const std::string_view& v>
    struct julia_type_factory<ValType<const std::string_view&, v>>
    {
        static inline jl_datatype_t* julia_type()
        {
            return apply_type(::jlcxx::julia_type("Val", jl_base_module), (jl_datatype_t*) jl_symbol(v.data()));
        }
    };

    template<typename T, T v>
    struct ConvertToCpp<ValType<T, v>, NoMappingTrait>
    {
        ValType<T, v> operator()(jl_datatype_t*) const
        {
            return ValType<T, v>();
        }
    };

    template<typename T, T v>
    struct ConvertToJulia<ValType<T, v>, NoMappingTrait>
    {
        jl_datatype_t* operator()(ValType<T, v>) const
        {
            static jl_datatype_t* type = apply_type(::jlcxx::julia_type("Val", jl_base_module), (jl_datatype_t*) ::jlcxx::box<T>(v));
            return type;
        }
    };

    template<const std::string_view& v>
    struct ConvertToJulia<ValType<const std::string_view&, v>, NoMappingTrait>
    {
        jl_datatype_t* operator()(ValType<const std::string_view&, v>) const
        {
            static jl_datatype_t* type = apply_type(::jlcxx::julia_type("Val", jl_base_module), (jl_datatype_t*) jl_symbol(v.data()));
            return type;
        }
    };
}


#define JLCXX_STATIC_SYM(sym) static constexpr auto sym = std::string_view(#sym)


#endif //KOKKOS_WRAPPER_JLCXX_VAL_H
