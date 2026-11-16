# Convergence diagnostics and posterior summaries.

"""
    effective_sample_size(x::AbstractVector) -> Float64
    effective_sample_size(chain::MCMCSamples, name::Symbol) -> Float64

The effective sample size of a chain of draws, `n / τ` with the integrated
autocorrelation time `τ` estimated by Geyer's initial monotone sequence
[Geyer1992](@cite): sums of autocorrelations at adjacent lags are added while
they are positive and made non-increasing. `NaN` for a constant chain.
"""
function effective_sample_size(x::AbstractVector{<:Real})
    n = length(x)
    n >= 4 || throw(ArgumentError("effective_sample_size needs at least 4 draws, got $n"))
    μ = mean(x)
    d = x .- μ
    c0 = sum(abs2, d) / n
    c0 > 0 || return NaN
    autocorr(k) = sum(d[i] * d[i + k] for i in 1:(n - k)) / (n * c0)
    τ = -1.0
    previous = Inf
    m = 0
    while 2m + 1 < n
        Γ = (m == 0 ? 1.0 : autocorr(2m)) + autocorr(2m + 1)
        Γ > 0 || break
        Γ = min(Γ, previous)
        τ += 2Γ
        previous = Γ
        m += 1
    end
    return n / max(τ, 1 / n)
end
effective_sample_size(chain::MCMCSamples, name::Symbol) = effective_sample_size(chain[name])

"""
    rhat(chains::AbstractVector{<:AbstractVector}) -> Float64
    rhat(chains::AbstractVector{MCMCSamples}, name::Symbol) -> Float64

The split potential scale reduction factor [Gelman2013](@cite) of several chains
of the same length: each chain is split in halves, and the between-half
variance is compared with the within-half variance. Values near one (below
about 1.01 to 1.05) suggest the chains have mixed.
"""
function rhat(chains::AbstractVector{<:AbstractVector{<:Real}})
    isempty(chains) && throw(ArgumentError("rhat needs at least one chain"))
    n = minimum(length, chains)
    all(c -> length(c) == n, chains) ||
        throw(ArgumentError("rhat needs chains of the same length"))
    h = n ÷ 2
    h >= 2 || throw(ArgumentError("rhat needs at least 4 draws per chain, got $n"))
    halves = [c[r] for c in chains for r in (1:h, (n - h + 1):n)]
    W = mean(var, halves)
    B = h * var(mean.(halves))
    return sqrt(((h - 1) / h * W + B / h) / W)
end
function rhat(chains::AbstractVector{MCMCSamples}, name::Symbol)
    return rhat([c[name] for c in chains])
end

function _weighted_quantile(x, w, q)
    order = sortperm(x)
    total = sum(w)
    acc = 0.0
    for i in order
        acc += w[i]
        acc >= q * total && return x[i]
    end
    return x[last(order)]
end

"""
    posterior_summary(chain::MCMCSamples; quantiles=(0.025, 0.5, 0.975)) -> NamedTuple
    posterior_summary(chains::AbstractVector{MCMCSamples}; quantiles) -> NamedTuple
    posterior_summary(post::ABCPosterior; quantiles) -> NamedTuple

For each parameter, a `NamedTuple` of its posterior `mean`, `sd`, the requested
`quantiles` (as `q`, the vector of their values) and the effective sample size
`ess`: from the draws of one chain or the pooled draws of several (summing the
chains' effective sample sizes, and adding `rhat`), or from the weighted
particles of an [`ABCPosterior`](@ref) (whose `ess` is `1 / sum(w .^ 2)`).
"""
function posterior_summary(chain::MCMCSamples; quantiles=(0.025, 0.5, 0.975))
    return NamedTuple{Tuple(chain.names)}(Tuple(begin
                                                    x = chain[k]
                                                    (mean=mean(x), sd=std(x),
                                                     q=[quantile(x, p) for p in quantiles],
                                                     ess=effective_sample_size(x))
                                                end
                                                for k in chain.names))
end
function posterior_summary(chains::AbstractVector{MCMCSamples};
                           quantiles=(0.025, 0.5, 0.975))
    isempty(chains) && throw(ArgumentError("posterior_summary needs at least one chain"))
    names = first(chains).names
    all(c -> c.names == names, chains) ||
        throw(ArgumentError("the chains must have the same parameters"))
    return NamedTuple{Tuple(names)}(Tuple(begin
                                              x = reduce(vcat, (c[k] for c in chains))
                                              (mean=mean(x), sd=std(x),
                                               q=[quantile(x, p) for p in quantiles],
                                               ess=sum(c -> effective_sample_size(c[k]),
                                                       chains),
                                               rhat=rhat(chains, k))
                                          end
                                          for k in names))
end
function posterior_summary(post::ABCPosterior; quantiles=(0.025, 0.5, 0.975))
    w = post.weights ./ sum(post.weights)
    return NamedTuple{Tuple(post.names)}(Tuple(begin
                                                   x = post[k]
                                                   m = sum(w .* x)
                                                   (mean=m,
                                                    sd=sqrt(sum(w .* (x .- m) .^ 2)),
                                                    q=[_weighted_quantile(x, w, p)
                                                       for p in quantiles],
                                                    ess=1 / sum(abs2, w))
                                               end
                                               for k in post.names))
end
