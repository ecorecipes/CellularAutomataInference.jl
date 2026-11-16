using Test
using Random
using Statistics
using Distributions
using LogDensityProblems: LogDensityProblems
using CellularAutomataCore
using CellularAutomataInference

@testset "CellularAutomataInference" begin
    include("test_priors.jl")
    include("test_mcmc.jl")
    include("test_abc.jl")
    include("test_diagnostics.jl")
    include("test_errors.jl")
    include("test_docstrings.jl")
end
