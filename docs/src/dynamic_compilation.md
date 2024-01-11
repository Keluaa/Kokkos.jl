```@meta
CurrentModule = Kokkos.DynamicCompilation
```

# Dynamic Compilation

Functions used to compile on demand the Kokkos functions needed.

This mechanism allows to create new methods without degrading performance (after method 
invalidation).

Only functions operating on views are dynamically compiled, the rest are compiled with the wrapper
library.

Compilation is not thread-safe and must be done by a single process (and single thread) at once.
This is ensured by [`compilation_lock`](@ref).

```@docs
@compile_and_call
compile_and_load
has_specialization
call_more_specific
clean_libs
compilation_lock
```
