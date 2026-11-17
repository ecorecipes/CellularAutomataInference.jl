# ABC-SMC through ABCdeZ: a normal mean whose posterior is known, reproducibility from a seeded
# rng, and a cellular automaton problem whose particles' distances must equal those of direct
# solves with the seeds derived from their parameters.

@testset "ABC-SMC" begin
    @testset "normal mean" begin
        n = 20
        ybar = 0.8
        simulate(θ, seed) = mean(θ.μ .+ randn(Xoshiro(seed), n))
        distance(sim, obs) = abs(sim - obs)
        post = abc_smc(simulate, (μ=Normal(0, 5),), distance, ybar; tolerance=0.01,
                       nparticles=400, rng=Xoshiro(11))
        @test post isa ABCPosterior && post.names == [:μ]
        @test post.tolerance <= 0.01 + 1e-12 && isfinite(post.logevidence)
        @test all(post.distances[post.weights .> 0] .<= post.tolerance)
        @test sum(post.weights) ≈ 1
        # posterior N(m, s²) with s² = 1 / (1/25 + n)
        s = sqrt(1 / (1 / 25 + n))
        m = n * ybar * s^2
        sm = posterior_summary(post)
        @test abs(sm.μ.mean - m) < 0.05
        @test abs(sm.μ.sd - s) < 0.2s
        @test first(post.tolerances) == Inf
        @test startswith(sprint(show, post), "ABCPosterior:")
        again = abc_smc(simulate, (μ=Normal(0, 5),), distance, ybar; tolerance=0.01,
                        nparticles=400, rng=Xoshiro(11))
        @test again.values == post.values && again.weights == post.weights
        @test_throws ArgumentError abc_smc(simulate, (μ=Normal(0, 5),), distance, ybar;
                                           tolerance=-1, rng=Xoshiro(1))
        @test_throws ArgumentError abc_smc(simulate, (μ=Normal(0, 5),),
                                           (s, o) -> -1.0, ybar; tolerance=0.1,
                                           nparticles=50, rng=Xoshiro(1))
        # distances tied at the tolerance all fail ABCdeZ's strict kernel: no particle lives
        @test_throws ABCFailureError abc_smc(simulate, (μ=Normal(0, 5),), (s, o) -> 1.0,
                                             0.0;
                                             tolerance=0.5, nparticles=50, rng=Xoshiro(2))
        # the inclusive kernel keeps them
        tied = abc_smc(simulate, (μ=Normal(0, 5),), (s, o) -> 1.0, 0.0; tolerance=0.5,
                       nparticles=50, rng=Xoshiro(2), strict=false, nsims_max=500)
        @test count(>(0), tied.weights) == 50 && tied.tolerance == 1.0
    end

    @testset "a cellular automaton problem" begin
        si = ProbabilisticRule(VonNeumann(1)) do c, n, ctx
            c == 1 && return (Int8(1) => 1.0,)
            q = 1 - (1 - ctx.p.β)^count(==(1), n)
            return (Int8(0) => 1 - q, Int8(1) => q)
        end
        u0 = zeros(Int8, 15, 15)
        u0[8, 8] = 1
        prob = CAProblem(si, u0, 8, (β=0.3,); seed=1)
        infected(sol) = count(==(1), final_state(sol))
        observed = infected(solve(remake(prob; seed=99); save_everystep=false))
        dist(sol, obs) = abs(infected(sol) - obs)
        post = abc_smc(prob, (β=Uniform(0, 1),), dist, observed; tolerance=2,
                       nparticles=100, rng=Xoshiro(12),
                       solve_kwargs=(save_everystep=false,))
        live = findall(>(0), post.weights)
        for i in live[1:5]
            θ = (β=post.values[i, 1],)
            sol = solve(remake(prob; p=θ,
                               seed=CellularAutomataInference._simulation_seed(θ));
                        save_everystep=false)
            @test dist(sol, observed) == post.distances[i]
        end
        @test 0.1 < posterior_summary(post).β.mean < 0.6
    end
end
