"""
    CellularAutomataInferenceError

Abstract supertype of every exception this package defines:
[`SupportError`](@ref) and [`ABCFailureError`](@ref). Invalid arguments
and keywords raise Base's `ArgumentError`, which is outside it; exceptions from
CellularAutomataCore (such as its `NoLikelihoodError`) pass through unchanged.
"""
abstract type CellularAutomataInferenceError <: Exception end

"""
    SupportError(what, value, msg)

Thrown when a sampler would start where the target has no density: a parameter
outside its prior's support, or a starting point whose log-likelihood is `-Inf`
or `NaN`. `what` names the parameter or quantity and `value` is the offending
value.
"""
struct SupportError <: CellularAutomataInferenceError
    what::String
    value::Any
    msg::String
end

function Base.showerror(io::IO, e::SupportError)
    return print(io, "SupportError: ", e.what, " = ", e.value, ": ", e.msg)
end

"""
    ABCFailureError(reached, target)

Thrown when [`abc_smc`](@ref) ends with no live particle, at tolerance
`reached` (the `target` was lower). Under the strict kernel this happens when
the particles' distances tie at the tolerance, as discrete summaries do: use
`strict=false`, or stop at a larger tolerance.
"""
struct ABCFailureError <: CellularAutomataInferenceError
    reached::Float64
    target::Float64
end

function Base.showerror(io::IO, e::ABCFailureError)
    return print(io, "ABCFailureError: no particle survived at tolerance ", e.reached,
                 " (target ", e.target,
                 "); with tied (discrete) distances use strict=false, or stop at a larger tolerance")
end
