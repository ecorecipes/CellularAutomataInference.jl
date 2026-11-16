# CellularAutomataInference.jl

Fitting cellular automata to data, for models built with `CellularAutomataCore.jl`.

Part of the ecorecipes cellular-automata ecosystem: `CellularAutomataCore.jl` → `CellularAutomataInference.jl`
(fitting), alongside `CategoricalCellularAutomata.jl` (applied category theory) and
`EcologicalCellularAutomata.jl` (model zoo).

## Two routes to a posterior

**With a likelihood.** When the probability of the data can be written down, sample the posterior by Markov
chain Monte Carlo. A [`PosteriorDensity`](@ref) combines a log-likelihood with priors (a `NamedTuple` of
distributions from Distributions.jl) on an unconstrained scale, as a LogDensityProblems object; wrapped in
`LogDensityModel`, it is sampled through the AbstractMCMC interface by [`AdaptiveMetropolis`](@ref)
[Haario2001](@cite) or by any other sampler of that interface. The core's `path_logpdf` gives the exact
likelihood of a path observed at every step, and its observation models give the probability of what a survey
or sensor reports.

Data rarely show every step: surveys are months apart, and an infection is seen only once symptoms appear.
The states in between are then latent, and the likelihood sums over them. Data augmentation
[Gibson1998](@cite) samples them instead, alongside the parameters: an [`AugmentedModel`](@ref) states the
joint density of data and latent variables and a move for the latent variables, and
[`DataAugmentation`](@ref) alternates those moves with adaptive Metropolis moves of the parameters.

```julia
model = AugmentedModel(loglik, latent_step!, priors, latent0)
chains = sample(rng, model, DataAugmentation(), MCMCThreads(), 20_000, 4)
posterior_summary(chains)
```

**By simulation.** When only simulation is practical, [`abc_smc`](@ref) runs approximate Bayesian computation
by sequential Monte Carlo with differential-evolution moves, through ABCdeZ.jl, on a simulator or directly on
a cellular automaton problem, and estimates the model evidence for comparing models. The core's
`abc_rejection` remains the simplest alternative.

```julia
post = abc_smc(prob, (β=Uniform(0, 1),), distance, observed; tolerance=2, rng=Xoshiro(1))
```

## Checking a fit

[`effective_sample_size`](@ref) [Geyer1992](@cite), split [`rhat`](@ref) [Gelman2013](@cite) and
[`posterior_summary`](@ref) summarise chains and weighted particles. Before fitting real data, fit data
simulated from known parameters and check that they are recovered.
