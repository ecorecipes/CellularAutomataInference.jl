# Samplers against posteriors known in closed form: a beta-binomial posterior, a correlated
# Gaussian given as a plain LogDensityProblems object, and, for data augmentation, a hierarchical
# normal model whose marginal posterior is normal.

struct Gaussian2
    μ::Vector{Float64}
    P::Matrix{Float64}
end
LogDensityProblems.logdensity(g::Gaussian2, x) = -0.5 * (x - g.μ)' * g.P * (x - g.μ)
LogDensityProblems.dimension(::Gaussian2) = 2
function LogDensityProblems.capabilities(::Type{Gaussian2})
    return LogDensityProblems.LogDensityOrder{0}()
end

within(x, target, se) = abs(x - target) < 5 * se

@testset "MCMC" begin
    @testset "adaptive Metropolis, beta-binomial" begin
        ℓ = PosteriorDensity(θ -> 7 * log(θ.p) + 13 * log(1 - θ.p), (p=Uniform(0, 1),))
        model = LogDensityModel(ℓ)
        chain = sample(Xoshiro(3), model, AdaptiveMetropolis(), 20_000; progress=false,
                       initial_params=(p=0.5,), discard_initial=2_000)
        @test chain isa MCMCSamples
        @test chain.names == [:p] && length(chain) == 20_000
        @test 0.2 < chain.acceptance < 0.8
        p = chain[:p]
        ess = effective_sample_size(p)
        # Beta(8, 14)
        m, s = 8 / 22, sqrt(8 * 14 / (22^2 * 23))
        @test within(mean(p), m, s / sqrt(ess))
        @test abs(std(p) - s) < 0.1s
        again = sample(Xoshiro(3), model, AdaptiveMetropolis(), 20_000; progress=false,
                       initial_params=(p=0.5,), discard_initial=2_000)
        @test again.values == chain.values
        @test startswith(sprint(show, chain), "MCMCSamples: 20000 iterations of p")
        @test constrain(ℓ, unconstrain(ℓ, (p=0.3,))).p ≈ 0.3
        # several chains, threaded
        chains = sample(Xoshiro(4), model, AdaptiveMetropolis(), MCMCThreads(), 5_000, 3;
                        progress=false)
        @test chains isa Vector{MCMCSamples} && length(chains) == 3
        @test rhat(chains, :p) < 1.05
        summary = posterior_summary(chains)
        @test abs(summary.p.mean - m) < 0.02 && summary.p.rhat < 1.05
    end

    @testset "adaptive Metropolis, correlated Gaussian" begin
        Σ = [1.0 0.9; 0.9 1.0]
        g = Gaussian2([1.0, -2.0], inv(Σ))
        chain = sample(Xoshiro(5), LogDensityModel(g), AdaptiveMetropolis(), 40_000;
                       progress=false, initial_params=[0.0, 0.0], discard_initial=5_000)
        @test chain.names == [:x1, :x2]
        X = chain.values
        @test within(mean(X[:, 1]), 1.0, 1 / sqrt(effective_sample_size(X[:, 1])))
        @test within(mean(X[:, 2]), -2.0, 1 / sqrt(effective_sample_size(X[:, 2])))
        @test isapprox(cov(X), Σ; atol=0.15)
        @test_throws ArgumentError sample(Xoshiro(1), LogDensityModel(g),
                                          AdaptiveMetropolis(),
                                          10; progress=false)
    end

    @testset "starting points" begin
        ℓ = PosteriorDensity(θ -> θ.p > 0.9 ? 0.0 : -Inf, (p=Uniform(0, 1),))
        @test_throws SupportError sample(Xoshiro(1), LogDensityModel(ℓ),
                                         AdaptiveMetropolis(), 10; progress=false,
                                         initial_params=(p=1.5,))
        @test_throws SupportError sample(Xoshiro(1), LogDensityModel(ℓ),
                                         AdaptiveMetropolis(), 10; progress=false,
                                         initial_params=(p=0.5,))
        @test_throws ArgumentError AdaptiveMetropolis(; adapt_after=1)
        @test_throws ArgumentError AdaptiveMetropolis(; init_scale=0)
    end

    @testset "data augmentation, hierarchical normal" begin
        # θ ~ N(0, 10²); z_i ~ N(θ, 1); y_i ~ N(z_i, 1): marginally y_i ~ N(θ, 2)
        rng = Xoshiro(6)
        n = 25
        y = 1.5 .+ sqrt(2) .* randn(rng, n)
        loglik(θ, z) = sum(logpdf.(Normal.(θ.θ, 1), z)) + sum(logpdf.(Normal.(z, 1), y))
        function gibbs!(rng, z, θ)
            for i in eachindex(z)
                z[i] = (θ.θ + y[i]) / 2 + sqrt(0.5) * randn(rng)
            end
            return nothing
        end
        model = AugmentedModel(loglik, gibbs!, (θ=Normal(0, 10),), zeros(n);
                               record=z -> mean(z))
        chain = sample(Xoshiro(7), model, DataAugmentation(; parameter_steps=2), 20_000;
                       progress=false, discard_initial=1_000, initial_params=(θ=0.0,))
        @test chain isa MCMCSamples && length(chain.records) == 20_000
        prec = 1 / 100 + n / 2
        m, s = (sum(y) / 2) / prec, sqrt(1 / prec)
        x = chain[:θ]
        @test within(mean(x), m, s / sqrt(effective_sample_size(x)))
        @test abs(std(x) - s) < 0.1s
        @test abs(mean(chain.records) - m) < 0.1
        # each chain has its own copy of the latent state
        chains = sample(Xoshiro(8), model, DataAugmentation(), MCMCThreads(), 2_000, 2;
                        progress=false)
        @test length(chains) == 2 && model.latent == zeros(n)
        @test_throws ArgumentError DataAugmentation(; parameter_steps=0)
        @test_throws SupportError sample(Xoshiro(1),
                                         AugmentedModel((θ, z) -> -Inf, gibbs!,
                                                        (θ=Normal(),), zeros(n)),
                                         DataAugmentation(), 5; progress=false)
    end
end
