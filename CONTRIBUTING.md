# Contributing to CellularAutomataInference.jl

## Local setup

The ecosystem packages depend on each other through `[sources]` entries that point at
sibling directories (`../OtherPackage.jl`). Clone every package you need side by side:

```sh
mkdir -p ca && cd ca
git clone https://github.com/ecorecipes/CellularAutomataCore.jl
git clone https://github.com/ecorecipes/CellularAutomataInference.jl
cd CellularAutomataInference.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

`Pkg.test()` is the only run that counts as "the suite passes".

## Style

- Julia ≥ 1.12. Every dependency, including stdlibs and test-only extras, has a `[compat]` entry.
- Format with JuliaFormatter using the repository `.JuliaFormatter.toml` (`style = "yas"`).
- `snake_case` for functions and variables, `CamelCase` for types, a leading underscore for internal helpers.
- Public functions have docstrings with a signature line, a one-sentence summary, and an example where practical.
- Exceptions are typed, carry the offending names, live in `src/errors.jl` and subtype
  `CellularAutomataInferenceError`; library code never calls a bare `error("...")`. Invalid arguments and keywords raise
  `ArgumentError`.
- Every optimised code path is tested against a slower reference implementation on small models.
