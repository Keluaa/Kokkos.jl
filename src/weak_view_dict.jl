
"""
    WeakViewDict

Very similar to `Base.WeakKeyDict`, but its keys are `Ptr{Cvoid}` and values are `WeakRef`s to `View`
objects. It is used to track view objects.

We cannot use `WeakKeyDict` as would need to hash the view, but this is inefficient and sometimes
impossible when the view is inaccessible. Any overload of `Base.hash` would go against what the
function operates on any `AbstractArray`. Therefore implementing a `WeakKeyDict` with `WeakRef`
values was the simplest path.
"""
mutable struct WeakViewDict <: AbstractDict{Ptr{Cvoid}, View}
    d::Dict{Ptr{Cvoid}, WeakRef}
    lock::ReentrantLock
    finalizer::Function
    dirty::Bool

    function WeakViewDict()
        wvd = new(Dict{Ptr{Cvoid}, WeakRef}(), ReentrantLock(), identity, false)
        wvd.finalizer = _ -> (wvd.dirty = true)
        return wvd
    end
end


Base.IteratorSize(::Type{WeakViewDict}) = Base.SizeUnknown()

Base.islocked(wvd::WeakViewDict) = islocked(wvd.lock)
Base.lock(wvd::WeakViewDict) = lock(wvd.lock)
Base.unlock(wvd::WeakViewDict) = unlock(wvd.lock)
Base.lock(f, wvd::WeakViewDict) = lock(f, wvd.lock)
Base.trylock(f, wvd::WeakViewDict) = trylock(f, wvd.lock)


function _cleanup_locked(wvd::WeakViewDict)
    wvd.dirty || return
    wvd.dirty = false
    idx = Base.skip_deleted_floor!(wvd.d)
    while idx != 0
        if wvd.d.vals[idx].value === nothing
            Base._delete!(wvd.d, idx)
        end
        idx = Base.skip_deleted(wvd.d, idx + 1)
    end
end


function Base.setindex!(wvd::WeakViewDict, view::View, key::Ptr{Cvoid})
    lock(wvd) do 
        _cleanup_locked(wvd)
        finalizer(wvd.finalizer, view)
        wvd.d[key] = WeakRef(view)
    end
    return wvd
end

Base.push!(wvd::WeakViewDict, view::View) = setindex!(wvd, view, view.cpp_object)


function Base.empty!(wvd::WeakViewDict)
    lock(wvd) do
        empty!(wvd.d)
    end
    return wvd
end


function Base.haskey(wvd::WeakViewDict, ptr::Ptr{Cvoid})
    lock(wvd) do
        return haskey(wvd.d, ptr)
    end
end

Base.haskey(wvd::WeakViewDict, view::View) = haskey(wvd, view.cpp_object)


function Base.iterate(wvd::WeakViewDict, state...)
    lock(wvd) do
        while true
            s = iterate(wvd.d, state...)
            s === nothing && return nothing
            kv, state = s
            v = kv[2].value
            GC.safepoint()
            v === nothing && continue
            return (v::View, state)
        end
    end
end
