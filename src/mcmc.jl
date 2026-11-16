# Markov chain Monte Carlo through the AbstractMCMC interface. A posterior is a LogDensityProblems
# object on the unconstrained scale, so it can be sampled here or by any other AbstractMCMC
# sampler that takes a LogDensityModel. AdaptiveMetropolis is the adaptive random-walk sampler of
# Haario et al.; DataAugmentation alternates a user's update of latent variables (unobserved
# infection times, say) with adaptive Metropolis steps for the parameters given them.

"""
    PosteriorDensity(loglik, priors)

A posterior density for the parameters named in `priors` (a `NamedTuple` of
continuous univariate distributions from Distributions.jl), with log-likelihood
`loglik(θ)` for a `NamedTuple` `θ`. It implements the LogDensityProblems
interface on the unconstrained scale: each parameter is mapped to the real line
by a bijection chosen from its prior's support (logit, log or identity), and the
density includes the log-Jacobian, so `logdensity` at `y` is the log posterior
density of `y`. [`constrain`](@ref) and [`unconstrain`](@ref) convert between
the scales. Wrap it in `AbstractMCMC.LogDensityModel` to sample it.
"""
struct PosteriorDensity{F,P<:NamedTuple}
    loglik::F
    priors::P
    function PosteriorDensity(loglik, priors::NamedTuple)
        _check_priors(priors)
        return new{typeof(loglik),typeof(priors)}(loglik, priors)
    end
end

LogDensityProblems.dimension(ℓ::PosteriorDensity) = length(ℓ.priors)
function LogDensityProblems.capabilities(::Type{<:PosteriorDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.logdensity(ℓ::PosteriorDensity, y::AbstractVector)
    lp = _logprior_unconstrained(ℓ.priors, y)
    isfinite(lp) || return -Inf
    ll = Float64(ℓ.loglik(_constrain(ℓ.priors, y)))
    return isnan(ll) ? -Inf : lp + ll
end

"""
    constrain(ℓ::PosteriorDensity, y) -> NamedTuple

The parameters at the point `y` of the unconstrained scale.
"""
constrain(ℓ::PosteriorDensity, y::AbstractVector) = _constrain(ℓ.priors, y)

"""
    unconstrain(ℓ::PosteriorDensity, θ::NamedTuple) -> Vector{Float64}

The point of the unconstrained scale for the parameters `θ`.
"""
unconstrain(ℓ::PosteriorDensity, θ::NamedTuple) = _unconstrain(ℓ.priors, θ)

# ------------------------------------------------------------------ adaptive Metropolis

"""
    AdaptiveMetropolis(; adapt_after=200, init_scale=0.1, epsilon=1e-6)

The adaptive Metropolis sampler [Haario2001](@cite), an
`AbstractMCMC.AbstractSampler` for an `AbstractMCMC.LogDensityModel`. For the
first `adapt_after` iterations the proposal adds independent normal steps with
standard deviation `init_scale` to every coordinate; afterwards it is normal
with covariance `2.38^2 / d * (C + epsilon * I)`, where `C` is the covariance
of the chain so far and `d` the dimension. On a [`PosteriorDensity`](@ref) the
chain moves on the unconstrained scale and records the parameters as a
`NamedTuple`.

```julia
model = AbstractMCMC.LogDensityModel(PosteriorDensity(loglik, priors))
chain = sample(rng, model, AdaptiveMetropolis(), 10_000; initial_params=θ0)
```

`sample` returns an [`MCMCSamples`](@ref); with `MCMCThreads()` and a number of
chains, a vector of them. Without `initial_params` a chain starts from a draw
of the prior.
"""
struct AdaptiveMetropolis <: AbstractMCMC.AbstractSampler
    adapt_after::Int
    init_scale::Float64
    epsilon::Float64
    function AdaptiveMetropolis(; adapt_after::Integer=200, init_scale::Real=0.1,
                                epsilon::Real=1e-6)
        adapt_after >= 2 || throw(ArgumentError("adapt_after must be at least 2"))
        init_scale > 0 || throw(ArgumentError("init_scale must be positive"))
        epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
        return new(Int(adapt_after), Float64(init_scale), Float64(epsilon))
    end
end

"""
    MetropolisTransition

One state of an [`AdaptiveMetropolis`](@ref) chain: `params` (a `NamedTuple`
on a [`PosteriorDensity`](@ref), otherwise the vector), its `logdensity` and
whether the move to it was `accepted`.
"""
struct MetropolisTransition{T}
    params::T
    logdensity::Float64
    accepted::Bool
end

mutable struct _AMState
    y::Vector{Float64}
    lp::Float64
    n::Int
    mean::Vector{Float64}
    M2::Matrix{Float64}
    accepted::Int
    steps::Int
end

function _AMState(y, lp)
    d = length(y)
    s = _AMState(copy(y), lp, 0, zeros(d), zeros(d, d), 0, 0)
    _am_record!(s)
    return s
end

# Welford's running mean and sum of squared deviations of the chain.
function _am_record!(s::_AMState)
    s.n += 1
    δ = s.y .- s.mean
    s.mean .+= δ ./ s.n
    s.M2 .+= δ * (s.y .- s.mean)'
    return s
end

function _am_proposal(rng, s::_AMState, sampler::AdaptiveMetropolis)
    d = length(s.y)
    s.n <= sampler.adapt_after && return s.y .+ sampler.init_scale .* randn(rng, d)
    C = Symmetric((2.38^2 / d) .* (s.M2 ./ (s.n - 1) + sampler.epsilon * I))
    return s.y .+ cholesky(C).L * randn(rng, d)
end

# One Metropolis step on the target `logdensity` (a function of y); returns whether it moved.
function _metropolis!(rng, logdensity, s::_AMState, sampler::AdaptiveMetropolis)
    y′ = _am_proposal(rng, s, sampler)
    lp′ = logdensity(y′)
    moved = !isnan(lp′) && log(rand(rng)) < lp′ - s.lp
    if moved
        s.y = y′
        s.lp = lp′
        s.accepted += 1
    end
    s.steps += 1
    _am_record!(s)
    return moved
end

_params(ℓ::PosteriorDensity, y) = constrain(ℓ, y)
_params(ℓ, y) = copy(y)

function _initial_point(rng, ℓ, initial_params)
    if initial_params === nothing
        ℓ isa PosteriorDensity ||
            throw(ArgumentError("initial_params is needed for a log density that is not a PosteriorDensity"))
        return unconstrain(ℓ, rand_prior(rng, ℓ.priors))
    elseif initial_params isa NamedTuple
        ℓ isa PosteriorDensity ||
            throw(ArgumentError("named initial_params need a PosteriorDensity"))
        _check_start(ℓ.priors, initial_params)
        return unconstrain(ℓ, initial_params)
    end
    length(initial_params) == LogDensityProblems.dimension(ℓ) ||
        throw(ArgumentError("initial_params needs $(LogDensityProblems.dimension(ℓ)) values"))
    return collect(Float64, initial_params)
end

function _check_finite_start(lp)
    isfinite(lp) ||
        throw(SupportError("the log density at the start", lp,
                           "start where the likelihood is positive"))
    return nothing
end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel,
                           sampler::AdaptiveMetropolis; initial_params=nothing, kwargs...)
    ℓ = model.logdensity
    y = _initial_point(rng, ℓ, initial_params)
    lp = Float64(LogDensityProblems.logdensity(ℓ, y))
    _check_finite_start(lp)
    return MetropolisTransition(_params(ℓ, y), lp, true), _AMState(y, lp)
end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel,
                           sampler::AdaptiveMetropolis, state::_AMState; kwargs...)
    ℓ = model.logdensity
    moved = _metropolis!(rng, y -> Float64(LogDensityProblems.logdensity(ℓ, y)), state,
                         sampler)
    return MetropolisTransition(_params(ℓ, state.y), state.lp, moved), state
end

# ------------------------------------------------------------------ chains

"""
    MCMCSamples

The output of `sample` with [`AdaptiveMetropolis`](@ref) or
[`DataAugmentation`](@ref): `names` (the parameters), `values` (one row per
kept iteration, one column per parameter, on the parameters' own scale),
`logdensity` (the log posterior density on the unconstrained scale, up to a
constant, or the joint density of parameters and latent variables under data
augmentation), `acceptance` (the share of parameter moves accepted) and
`records` (what an [`AugmentedModel`](@ref)'s `record` returned at each kept
iteration; empty otherwise). `chain[:β]` is the column of parameter `β`.
"""
struct MCMCSamples
    names::Vector{Symbol}
    values::Matrix{Float64}
    logdensity::Vector{Float64}
    acceptance::Float64
    records::Vector{Any}
end

Base.getindex(s::MCMCSamples, name::Symbol) = s.values[:, _column(s, name)]
Base.length(s::MCMCSamples) = size(s.values, 1)
function _column(s::MCMCSamples, name::Symbol)
    j = findfirst(==(name), s.names)
    j === nothing && throw(ArgumentError("no parameter $name; the chain has $(s.names)"))
    return j
end
function Base.show(io::IO, s::MCMCSamples)
    return print(io, "MCMCSamples: ", length(s), " iterations of ", join(s.names, ", "),
                 "; acceptance ", round(s.acceptance; digits=3))
end

_names(p::NamedTuple) = collect(keys(p))
_names(p::AbstractVector) = [Symbol("x", i) for i in eachindex(p)]

function _bundle(ts, records)
    isempty(ts) && throw(ArgumentError("no samples to bundle"))
    vals = reduce(vcat, (collect(Float64, values(t.params))' for t in ts))
    return MCMCSamples(_names(first(ts).params), vals, [t.logdensity for t in ts],
                       count(t -> t.accepted, ts) / length(ts), records)
end

for C in (Any, MCMCSamples)
    @eval function AbstractMCMC.bundle_samples(ts::Vector{<:MetropolisTransition},
                                               model::AbstractMCMC.LogDensityModel,
                                               sampler::AdaptiveMetropolis, state,
                                               ::Type{$C}; kwargs...)
        return _bundle(ts, Any[])
    end
end

# ------------------------------------------------------------------ data augmentation

"""
    AugmentedModel(loglik, latent_step!, priors, latent; record=nothing)

A model with parameters (named in `priors`, a `NamedTuple` of continuous
univariate distributions) and latent variables (unobserved event times, hidden
states), for sampling by data augmentation [Gibson1998](@cite) with
[`DataAugmentation`](@ref). `loglik(θ, z)` is the joint log-density of the data
and the latent variables `z` given the parameters `θ`; `latent_step!(rng, z, θ)`
updates `z` in place by any move that leaves the conditional distribution of `z`
given `θ` and the data invariant (Metropolis-Hastings or Gibbs updates of each
latent variable, say). `latent` is the initial latent state, copied for each
chain. `record(z)`, if given, is stored at every kept iteration (copy anything
mutable).
"""
struct AugmentedModel{F,L,P<:NamedTuple,Z,R} <: AbstractMCMC.AbstractModel
    loglik::F
    latent_step!::L
    priors::P
    latent::Z
    record::R
    function AugmentedModel(loglik, latent_step!, priors::NamedTuple, latent;
                            record=nothing)
        _check_priors(priors)
        return new{typeof(loglik),typeof(latent_step!),typeof(priors),typeof(latent),
                   typeof(record)}(loglik, latent_step!, priors, latent, record)
    end
end

"""
    DataAugmentation(; parameter_steps=1, adapt_after=200, init_scale=0.1, epsilon=1e-6)

The sampler for an [`AugmentedModel`](@ref): each iteration updates the latent
variables with the model's `latent_step!`, then makes `parameter_steps`
[`AdaptiveMetropolis`](@ref) moves of the parameters given the latent
variables (the other keywords configure them). The chain targets the joint
posterior of parameters and latent variables; its parameter columns are a
sample from their marginal posterior.
"""
struct DataAugmentation <: AbstractMCMC.AbstractSampler
    metropolis::AdaptiveMetropolis
    parameter_steps::Int
    function DataAugmentation(; parameter_steps::Integer=1, kwargs...)
        parameter_steps >= 1 || throw(ArgumentError("parameter_steps must be at least 1"))
        return new(AdaptiveMetropolis(; kwargs...), Int(parameter_steps))
    end
end

"""
    AugmentedTransition

One state of a [`DataAugmentation`](@ref) chain: `params`, the joint
`logdensity` of parameters and latent variables, whether any parameter move
was `accepted`, and the model's `record` of the latent variables (or
`nothing`).
"""
struct AugmentedTransition{T,R}
    params::T
    logdensity::Float64
    accepted::Bool
    record::R
end

mutable struct _DAState{Z}
    z::Z
    am::_AMState
end

function _joint(model::AugmentedModel, y, z)
    lp = _logprior_unconstrained(model.priors, y)
    isfinite(lp) || return -Inf
    ll = Float64(model.loglik(_constrain(model.priors, y), z))
    return isnan(ll) ? -Inf : lp + ll
end

_record(model::AugmentedModel, z) = model.record === nothing ? nothing : model.record(z)

function AbstractMCMC.step(rng::AbstractRNG, model::AugmentedModel,
                           sampler::DataAugmentation; initial_params=nothing, kwargs...)
    z = deepcopy(model.latent)
    if initial_params === nothing
        y = _unconstrain(model.priors, rand_prior(rng, model.priors))
    else
        initial_params isa NamedTuple ||
            throw(ArgumentError("initial_params must be a NamedTuple of the parameters"))
        _check_start(model.priors, initial_params)
        y = _unconstrain(model.priors, initial_params)
    end
    lp = _joint(model, y, z)
    _check_finite_start(lp)
    state = _DAState(z, _AMState(y, lp))
    return AugmentedTransition(_constrain(model.priors, y), lp, true, _record(model, z)),
           state
end

function AbstractMCMC.step(rng::AbstractRNG, model::AugmentedModel,
                           sampler::DataAugmentation, state::_DAState; kwargs...)
    am = state.am
    model.latent_step!(rng, state.z, _constrain(model.priors, am.y))
    target = y -> _joint(model, y, state.z)
    am.lp = target(am.y)
    moved = false
    for _ in 1:(sampler.parameter_steps)
        moved |= _metropolis!(rng, target, am, sampler.metropolis)
    end
    return AugmentedTransition(_constrain(model.priors, am.y), am.lp, moved,
                               _record(model, state.z)), state
end

for C in (Any, MCMCSamples)
    @eval function AbstractMCMC.bundle_samples(ts::Vector{<:AugmentedTransition},
                                               model::AugmentedModel,
                                               sampler::DataAugmentation, state,
                                               ::Type{$C}; kwargs...)
        return _bundle(ts, Any[t.record for t in ts])
    end
end
