# Approximate Bayesian computation through ABCdeZ.jl: sequential Monte Carlo with
# differential-evolution moves and an estimate of the model evidence. This file supplies the
# glue: named priors, a simulator or a cellular automaton problem with a seed for every
# simulation, and a named, weighted posterior.

"""
    ABCPosterior

The result of [`abc_smc`](@ref): `names` (the parameters), `values` (one row per
particle, one column per parameter), `weights` (normalised, zero for particles
that died; weight the particles when summarising), `distances` (each particle's
distance from the observations), `tolerance` (the final ABC tolerance),
`logevidence` (the estimated log model evidence at that tolerance, for comparing
models fitted to the same data with the same distance) and `tolerances` (the
sequence of tolerances, starting at `Inf`). `post[:β]` is the column of `β`.
"""
struct ABCPosterior
    names::Vector{Symbol}
    values::Matrix{Float64}
    weights::Vector{Float64}
    distances::Vector{Float64}
    tolerance::Float64
    logevidence::Float64
    tolerances::Vector{Float64}
end

function Base.getindex(r::ABCPosterior, name::Symbol)
    j = findfirst(==(name), r.names)
    j === nothing &&
        throw(ArgumentError("no parameter $name; the posterior has $(r.names)"))
    return r.values[:, j]
end
Base.length(r::ABCPosterior) = size(r.values, 1)
function Base.show(io::IO, r::ABCPosterior)
    return print(io, "ABCPosterior: ", count(>(0), r.weights), " live particles of ",
                 length(r), " for ", join(r.names, ", "), "; tolerance ",
                 round(r.tolerance; sigdigits=3), ", log evidence ",
                 round(r.logevidence; sigdigits=4))
end

# A simulation's seed is a function of its parameters, so the distance of a parameter set does
# not depend on which thread computes it or when.
_simulation_seed(θ::NamedTuple) = hash(Tuple(Float64.(values(θ))), 0x434149_4e46_4552)

"""
    abc_smc(simulate, priors, distance, observed; tolerance, nparticles=500, rng,
            parallel=false, strict=true, verbose=false, kwargs...) -> ABCPosterior
    abc_smc(prob::CAProblem, priors, distance, observed; parameters=identity,
            alg=Synchronous(), solve_kwargs=(;), kwargs...) -> ABCPosterior

Approximate Bayesian computation by sequential Monte Carlo with
differential-evolution moves, through ABCdeZ.jl's `abcdesmc!`. `priors` is a
`NamedTuple` of univariate distributions from Distributions.jl;
`simulate(θ, seed)` simulates data for the parameters `θ` (a `NamedTuple`)
with the random seed `seed`, and `distance(simulated, observed)` is a
nonnegative number. The tolerance falls adaptively until it reaches `tolerance`
(or ABCdeZ's other stopping rules apply; extra keywords such as `nsims_max`,
`α` and `Kmcmc` pass through to `abcdesmc!`).

The second form fits a cellular automaton problem: each simulation is
`solve(remake(prob; p=parameters(θ), seed), alg; solve_kwargs...)`, and the
distance compares that solution with `observed`.

Each simulation's seed is a hash of its parameter values, so a particle's
distance does not depend on the order or thread in which it was computed.
A particle is kept while its distance is below the tolerance (`strict`, ABCdeZ's
default) or at most the tolerance (`strict=false`). With discrete summaries,
such as counts, many particles share a distance, and under the strict kernel a
tolerance equal to that shared value removes them all: use `strict=false`
there. If no particle survives, `abc_smc` raises [`ABCFailureError`](@ref).
`rng` drives the sampler itself: a run is reproducible with `parallel=false`
and a seeded `rng`; with `parallel=true` the threads share `rng`, which must be
thread-safe, and runs are not reproducible.
"""
function abc_smc(simulate, priors::NamedTuple, distance, observed; tolerance::Real,
                 nparticles::Integer=500, rng::AbstractRNG, parallel::Bool=false,
                 strict::Bool=true, verbose::Bool=false, kwargs...)
    _check_priors(priors)
    tolerance >= 0 || throw(ArgumentError("tolerance must be nonnegative, got $tolerance"))
    names = keys(priors)
    prior = ABCdeZ.Factored(values(priors)...)
    dist! = function (θ, ve)
        nt = NamedTuple{names}(Float64.(Tuple(θ)))
        d = Float64(distance(simulate(nt, _simulation_seed(nt)), observed))
        d >= 0 || throw(ArgumentError("distances must be nonnegative, got $d at $nt"))
        return d, nothing
    end
    kernel = strict ? ABCdeZ.IndicatorStrict0toϵ : ABCdeZ.Indicator0toϵ
    r = ABCdeZ.abcdesmc!(prior, dist!, Float64(tolerance), nothing;
                         nparticles=Int(nparticles), rng, parallel, verbose,
                         verboseout=true, ABCk=kernel, kwargs...)
    any(>(0), r.Wns) || throw(ABCFailureError(Float64(r.ϵ), Float64(tolerance)))
    vals = reduce(vcat, (collect(Float64, Tuple(p))' for p in r.P))
    return ABCPosterior(collect(names), vals, collect(Float64, r.Wns),
                        collect(Float64, r.C), Float64(r.ϵ), Float64(r.logZ),
                        collect(Float64, r.ϵs))
end

function abc_smc(prob::CAProblem, priors::NamedTuple, distance, observed;
                 parameters=identity, alg=Synchronous(), solve_kwargs=NamedTuple(),
                 kwargs...)
    simulate = (θ, seed) -> solve(remake(prob; p=parameters(θ), seed=seed), alg;
                                  solve_kwargs...)
    return abc_smc(simulate, priors, distance, observed; kwargs...)
end
