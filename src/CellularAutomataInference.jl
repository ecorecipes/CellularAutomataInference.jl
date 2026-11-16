"""
    CellularAutomataInference

Fitting cellular automata to data, for models built with
`CellularAutomataCore.jl`.

- Priors are `NamedTuple`s of continuous univariate distributions from
  Distributions.jl ([`logprior`](@ref), [`rand_prior`](@ref)).
- Likelihood-based fitting goes through the AbstractMCMC interface. A
  [`PosteriorDensity`](@ref) is a LogDensityProblems object on an
  unconstrained scale, so `AbstractMCMC.LogDensityModel` wraps it for
  [`AdaptiveMetropolis`](@ref) or any other sampler of that interface; an
  [`AugmentedModel`](@ref) with [`DataAugmentation`](@ref) samples parameters
  together with latent variables, such as the unobserved infection times
  between surveys. The core's `path_logpdf` gives the exact likelihood of a
  fully observed path, and its observation models score what a survey or
  sensor reports.
- Simulation-based fitting is approximate Bayesian computation by sequential
  Monte Carlo through ABCdeZ.jl ([`abc_smc`](@ref)), with model evidence for
  comparing models.
- [`effective_sample_size`](@ref), [`rhat`](@ref) and
  [`posterior_summary`](@ref) summarise the results.

This package depends on `CellularAutomataCore.jl` and on no other package of
the ecosystem.
"""
module CellularAutomataInference

using ABCdeZ: ABCdeZ
using AbstractMCMC: AbstractMCMC, LogDensityModel, MCMCSerial, MCMCThreads, sample
using CellularAutomataCore: CAProblem, Synchronous, remake, solve
using Distributions: ContinuousUnivariateDistribution, logpdf
using LinearAlgebra: I, Symmetric, cholesky
using LogDensityProblems: LogDensityProblems
using Random: AbstractRNG
using Statistics: mean, quantile, std, var

# Priors
export logprior, rand_prior
# Likelihood-based fitting through AbstractMCMC
export PosteriorDensity, constrain, unconstrain
export AdaptiveMetropolis, MetropolisTransition, MCMCSamples
export AugmentedModel, DataAugmentation, AugmentedTransition
export sample, MCMCSerial, MCMCThreads, LogDensityModel
# Simulation-based fitting through ABCdeZ
export abc_smc, ABCPosterior
# Diagnostics
export effective_sample_size, rhat, posterior_summary
# Exceptions
export CellularAutomataInferenceError, SupportError

include("errors.jl")
include("priors.jl")
include("mcmc.jl")
include("abc.jl")
include("diagnostics.jl")

end # module
