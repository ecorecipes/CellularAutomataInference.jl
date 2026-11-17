# API Reference

## Priors

Priors are `NamedTuple`s of continuous univariate distributions from Distributions.jl, independent of each
other.

```@docs
logprior
rand_prior
```

## Likelihood-based fitting

`sample`, `MCMCSerial`, `MCMCThreads` and `LogDensityModel` are re-exported from AbstractMCMC.jl.

```@docs
PosteriorDensity
constrain
unconstrain
AdaptiveMetropolis
MetropolisTransition
AugmentedModel
DataAugmentation
AugmentedTransition
MCMCSamples
```

## Simulation-based fitting

```@docs
abc_smc
ABCPosterior
```

## Diagnostics

```@docs
effective_sample_size
rhat
posterior_summary
```

## Exceptions

```@docs
CellularAutomataInferenceError
SupportError
ABCFailureError
```

## Module

```@docs
CellularAutomataInference
```
