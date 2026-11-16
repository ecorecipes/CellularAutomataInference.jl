# Priors and the unconstrained scale: the transforms invert, their log-Jacobians agree with
# finite differences, and each prior's density on the unconstrained scale integrates to one.

const CAI = CellularAutomataInference

@testset "priors" begin
    priors = (a=Uniform(1, 3), b=Exponential(2), c=Normal(0, 1), d=LogNormal(0, 0.5),
              e=truncated(Normal(); upper=2))
    θ = (a=2.5, b=0.7, c=-0.3, d=1.4, e=1.1)
    @test logprior(priors, θ) ≈ sum(logpdf(priors[k], θ[k]) for k in keys(priors))
    @test logprior(priors, (c=-0.3, a=2.5, b=0.7, e=1.1, d=1.4)) ≈ logprior(priors, θ)
    @test logprior(priors, merge(θ, (a=4.0,))) == -Inf
    @test rand_prior(Xoshiro(1), priors) == rand_prior(Xoshiro(1), priors)
    @test all(k -> insupport(priors[k], rand_prior(Xoshiro(2), priors)[k]), keys(priors))

    for (k, d) in pairs(priors)
        x = θ[k]
        y = CAI._link(d, x)
        @test CAI._invlink(d, y) ≈ x
        h = 1e-6
        fd = (CAI._invlink(d, y + h) - CAI._invlink(d, y - h)) / 2h
        @test CAI._logjac(d, y) ≈ log(abs(fd)) atol = 1e-6
        # the density on the unconstrained scale integrates to one
        ys = range(-40, 40; length=200_001)
        f = [exp(logpdf(d, CAI._invlink(d, t)) + CAI._logjac(d, t)) for t in ys]
        @test sum(f) * step(ys) ≈ 1 atol = 1e-4
    end
    y = CAI._unconstrain(priors, θ)
    @test all(isapprox.(values(CAI._constrain(priors, y)), values(θ)))
    @test CAI._logprior_unconstrained(priors, y) ≈
          logprior(priors, θ) + sum(CAI._logjac(priors[i], y[i]) for i in 1:5)

    @test_throws ArgumentError logprior((a=Poisson(2),), (a=1.0,))
    @test_throws ArgumentError logprior((a=Normal(),), (b=1.0,))
    @test_throws ArgumentError rand_prior(Xoshiro(1), NamedTuple())
end
