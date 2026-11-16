# CellularAutomataInference.jl

Fitting cellular automata to data, for models built with `CellularAutomataCore.jl`.

## Place in the ecosystem

Dependency order (arrows = depends on): CellularAutomataInference → CellularAutomataCore. Within the
ecosystem this package depends on `CellularAutomataCore.jl` only, through a `[sources]` path to the sibling
directory. Its other dependencies are ABCdeZ (ABC-SMC), AbstractMCMC (the sampler interface),
Distributions (priors), LogDensityProblems (densities), LinearAlgebra, Random and Statistics.

The shared bibliography lives in the development workspace that holds the packages side by side; the
`references.bib` files here are synced copies.

## Invariants that must not be broken

- No Catlab, GATlab or ACSets, and no dependency on the zoo or the categorical package.
- Samplers follow AbstractMCMC: a sampler subtypes `AbstractMCMC.AbstractSampler` and implements
  `AbstractMCMC.step` (initial and subsequent), and `bundle_samples` returns an `MCMCSamples`. Posteriors are
  LogDensityProblems objects, so other AbstractMCMC samplers can use them.
- Priors are `NamedTuple`s of continuous univariate distributions; samplers move on the unconstrained scale,
  with the log-Jacobian in the density.
- ABC goes through ABCdeZ; every simulation's seed is a hash of its parameters, so distances do not depend on
  thread scheduling. Runs are reproducible with `parallel=false` and a seeded `rng`.
- Every sampler is tested against a posterior known in closed form; diagnostics against formulas written out
  in the tests.

## Layout

- `src/CellularAutomataInference.jl`: module docstring, exports, includes.
- `src/errors.jl`: the root `CellularAutomataInferenceError` and every exception under it.
- `src/priors.jl`: `logprior`, `rand_prior`, the transforms to the unconstrained scale.
- `src/mcmc.jl`: `PosteriorDensity`, `AdaptiveMetropolis`, `AugmentedModel`, `DataAugmentation`,
  `MCMCSamples`.
- `src/abc.jl`: `abc_smc`, `ABCPosterior`. `src/diagnostics.jl`: `effective_sample_size`, `rhat`,
  `posterior_summary`.
- `test/`: one file per source file, `test_errors.jl`, `test_docstrings.jl`; seeded RNGs only.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the test suite
julia --project=docs docs/make.jl --no-deploy                     # build docs locally
```

## Files not to edit by hand

- `docs/src/references.bib` and `vignettes/references.bib` are copied from the workspace
  `docs/references.bib` by `scripts/sync_references.jl`; edit the workspace copy.

## Style

JuliaFormatter `yas`; docstrings on every exported name, which `test/test_docstrings.jl` enforces; the docs
build is strict (no `warnonly`), so a docstring left out of the manual or a broken `@ref` fails it; typed
exceptions in `src/errors.jl` under `CellularAutomataInferenceError`, `ArgumentError` for invalid arguments
and keywords; seeded RNGs only in tests; no emojis in code or docs.
