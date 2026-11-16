# Diagnostics: effective sample sizes against independent draws and the known autocorrelation
# time of an AR(1) chain; split R-hat against the formula of Gelman et al. written out here;
# weighted summaries against replicated draws.

function hand_rhat(chains)
    n = length(first(chains))
    h = n ÷ 2
    halves = [c[r] for c in chains for r in (1:h, (n - h + 1):n)]
    m = length(halves)
    means = [sum(x) / h for x in halves]
    grand = sum(means) / m
    B = h / (m - 1) * sum((means .- grand) .^ 2)
    W = sum(sum((x .- mx) .^ 2) / (h - 1) for (x, mx) in zip(halves, means)) / m
    return sqrt(((h - 1) / h * W + B / h) / W)
end

@testset "diagnostics" begin
    rng = Xoshiro(21)
    x = randn(rng, 10_000)
    @test 8_500 < effective_sample_size(x) < 11_500
    ρ = 0.8
    n = 100_000
    a = zeros(n)
    for t in 2:n
        a[t] = ρ * a[t - 1] + sqrt(1 - ρ^2) * randn(rng)
    end
    @test abs(effective_sample_size(a) / (n * (1 - ρ) / (1 + ρ)) - 1) < 0.15
    @test isnan(effective_sample_size(fill(1.0, 10)))
    @test_throws ArgumentError effective_sample_size([1.0, 2.0])

    chains = [randn(rng, 1_000) for _ in 1:4]
    @test rhat(chains) ≈ hand_rhat(chains)
    @test rhat(chains) < 1.01
    shifted = [chains[1:3]; [chains[4] .+ 1]]
    @test rhat(shifted) ≈ hand_rhat(shifted)
    @test rhat(shifted) > 1.1
    @test_throws ArgumentError rhat([randn(10), randn(11)])

    # weighted particles summarise like replicated draws
    vals = [1.0, 2.0, 3.0, 4.0]
    w = [1, 2, 0, 1] ./ 4
    post = ABCPosterior([:a], reshape(vals, :, 1), w, zeros(4), 0.0, 0.0, [Inf, 0.0])
    rep = [1.0, 2.0, 2.0, 4.0]
    s = posterior_summary(post; quantiles=(0.25, 0.5, 1.0))
    @test s.a.mean ≈ mean(rep)
    @test s.a.sd ≈ sqrt(mean((rep .- mean(rep)) .^ 2))
    @test s.a.q == [1.0, 2.0, 4.0]
    @test s.a.ess ≈ 1 / sum(abs2, w)
    @test post[:a] == vals
    @test_throws ArgumentError post[:b]
end
