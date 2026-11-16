# Priors are NamedTuples of continuous univariate distributions from Distributions.jl, one per
# parameter, taken as independent. Samplers move on an unconstrained scale: each parameter is
# mapped to the real line by a bijection chosen from its prior's support (logit for a bounded
# interval, log for a half-line, identity for the whole line), and densities carry the
# log-Jacobian of the inverse map.

"""
    logprior(priors::NamedTuple, θ::NamedTuple) -> Float64

The log prior density of the parameters `θ`: the sum over parameters of
`logpdf(priors[k], θ[k])`, for a `NamedTuple` of continuous univariate
distributions from Distributions.jl. `θ` must have exactly the parameters of
`priors`, in any order.
"""
function logprior(priors::NamedTuple, θ::NamedTuple)
    _check_priors(priors)
    _check_names(priors, θ)
    return sum(k -> Float64(logpdf(priors[k], θ[k])), keys(priors); init=0.0)
end

"""
    rand_prior(rng, priors::NamedTuple) -> NamedTuple

One draw of every parameter from its prior, made with the random number
generator `rng`.
"""
function rand_prior(rng::AbstractRNG, priors::NamedTuple)
    _check_priors(priors)
    return map(d -> Float64(rand(rng, d)), priors)
end

function _check_priors(priors::NamedTuple)
    isempty(priors) && throw(ArgumentError("priors must name at least one parameter"))
    for (k, d) in pairs(priors)
        d isa ContinuousUnivariateDistribution ||
            throw(ArgumentError("the prior of $k must be a continuous univariate distribution, got $(typeof(d))"))
    end
    return nothing
end

function _check_names(priors::NamedTuple, θ::NamedTuple)
    Set(keys(θ)) == Set(keys(priors)) ||
        throw(ArgumentError("the parameters $(keys(θ)) do not match the priors' $(keys(priors))"))
    return nothing
end

_log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

# The bijection of a prior's support to the real line, its inverse, and the log-Jacobian of the
# inverse (log dx/dy).
function _link(d, x)
    lo, hi = minimum(d), maximum(d)
    if isfinite(lo) && isfinite(hi)
        return log(x - lo) - log(hi - x)
    elseif isfinite(lo)
        return log(x - lo)
    elseif isfinite(hi)
        return log(hi - x)
    end
    return Float64(x)
end
function _invlink(d, y)
    lo, hi = minimum(d), maximum(d)
    if isfinite(lo) && isfinite(hi)
        return lo + (hi - lo) / (1 + exp(-y))
    elseif isfinite(lo)
        return lo + exp(y)
    elseif isfinite(hi)
        return hi - exp(y)
    end
    return Float64(y)
end
function _logjac(d, y)
    lo, hi = minimum(d), maximum(d)
    if isfinite(lo) && isfinite(hi)
        return log(hi - lo) - _log1pexp(-y) - _log1pexp(y)
    elseif isfinite(lo) || isfinite(hi)
        return Float64(y)
    end
    return 0.0
end

_in_support(d, x) = isfinite(x) && minimum(d) < x < maximum(d)

function _unconstrain(priors::NamedTuple, θ::NamedTuple)
    _check_names(priors, θ)
    return [Float64(_link(priors[k], θ[k])) for k in keys(priors)]
end
function _constrain(priors::NamedTuple{K}, y::AbstractVector) where {K}
    return NamedTuple{K}(ntuple(i -> Float64(_invlink(priors[i], y[i])), length(K)))
end
function _logprior_unconstrained(priors::NamedTuple, y::AbstractVector)
    s = 0.0
    for (i, d) in enumerate(values(priors))
        s += logpdf(d, _invlink(d, y[i])) + _logjac(d, y[i])
    end
    return s
end

function _check_start(priors::NamedTuple, θ0::NamedTuple)
    _check_names(priors, θ0)
    for k in keys(priors)
        _in_support(priors[k], θ0[k]) ||
            throw(SupportError(string(k), θ0[k],
                               "the starting value must lie strictly inside its prior's support"))
    end
    return nothing
end
