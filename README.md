# CellularAutomataInference.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fitting cellular automata to data, for models built with `CellularAutomataCore.jl`.

Part of the ecorecipes cellular-automata ecosystem: `CellularAutomataCore.jl` → `CellularAutomataInference.jl`
(fitting), alongside `CategoricalCellularAutomata.jl` (applied category theory) and
`EcologicalCellularAutomata.jl` (model zoo).

## Features

- Priors as `NamedTuple`s of distributions from Distributions.jl, sampled on an unconstrained scale.
- Likelihood-based fitting through the AbstractMCMC interface: a `PosteriorDensity` is a LogDensityProblems
  object, sampled by `AdaptiveMetropolis` or any other AbstractMCMC sampler; `AugmentedModel` and
  `DataAugmentation` sample parameters together with latent variables such as unobserved infection times.
- Simulation-based fitting: approximate Bayesian computation by sequential Monte Carlo through ABCdeZ.jl, on a
  simulator or a cellular automaton problem, with model evidence.
- Diagnostics: effective sample sizes, split R-hat and posterior summaries of chains and weighted particles.

## Installation

The ecosystem packages are not registered, and this package finds its siblings through `[sources]` paths, so
clone them side by side:

```sh
mkdir -p ca && cd ca
git clone https://github.com/ecorecipes/CellularAutomataCore.jl
git clone https://github.com/ecorecipes/CellularAutomataInference.jl
cd CellularAutomataInference.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Requires Julia ≥ 1.12.

## References

The full bibliography for the ecosystem is `docs/src/references.bib`; the rendered list is the References page
of the documentation.
