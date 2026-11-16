"""
    CellularAutomataInferenceError

Abstract supertype of every exception this package defines:
[`SupportError`](@ref). Invalid arguments
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
