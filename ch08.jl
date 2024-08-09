# 8.3.
# EM アルゴリズムを実装して、適当な確率密度関数を推定せよ

using Distributions
using Plots

# 訓練標本数
const N = 10^4
# 混合数
const M = 3

function gaussuan_mixture(μ, σ, w)::MixtureModel
    return MixtureModel(
        [Normal(μ[i], σ[i]) for i in 1:M],
        w
    )
end

p = gaussuan_mixture([1.0, 3.0, 4.0], [0.5, 0.4, 0.2], [0.2, 0.5, 0.3])

x = rand(p, N)

# E-step
function e_step(x, ŵ, μ̂, σ̂)
    ϕ(x; μ, σ) = pdf(Normal(μ, σ), x)
    η̂ = zeros(N, M)

    for n in 1:N
        for l in 1:M
            η̂[n, l] = (ŵ[l] * ϕ(x[n]; μ=μ̂[l], σ=σ̂[l])) / sum(ŵ[k] * ϕ(x[n]; μ=μ̂[k], σ=σ̂[k]) for k in 1:M)
        end
    end

    return η̂
end


# M-step
function m_step(x, η̂)
    μ̂ = zeros(M)
    σ̂ = zeros(M)
    ŵ = zeros(M)

    for l in 1:M
        ŵ[l] = sum(η̂[i, l] for i in 1:N) / N
        μ̂[l] = sum(η̂[i, l] * x[i] for i in 1:N) / sum(η̂[i, l] for i in 1:N)
        σ̂[l] = sqrt(sum(η̂[i, l] * (x[i] - μ̂[l])^2 for i in 1:N) / sum(η̂[i, l] for i in 1:N))
    end

    return μ̂, σ̂, ŵ
end

struct EMState
    μ::Vector{Float64}
    σ²::Vector{Float64}
    w::Vector{Float64}
    log_likelihood::Float64
end


function em_algorithm(x; μ̂₁, σ̂₁, ŵ₁, max_iter=100)
    history = EMState[]
    μ̂ = μ̂₁
    σ̂ = σ̂₁
    ŵ = ŵ₁

    for _ in 1:max_iter
        η̂ = e_step(x, ŵ, μ̂, σ̂)
        μ̂, σ̂, ŵ = m_step(x, η̂)
        m = gaussuan_mixture(μ̂, σ̂, ŵ)
        log_likelihood = loglikelihood(m, x)
        push!(history, EMState(μ̂, σ̂, ŵ, log_likelihood))
    end

    return history
end


history = em_algorithm(
    x,
    μ̂₁=[1.0, 2.0, 3.0],
    σ̂₁=[1.0, 2.0, 3.0],
    ŵ₁=[1 / 3, 1 / 3, 1 / 3]
)

μ̂ = history[end].μ
σ̂ = history[end].σ²
ŵ = history[end].w


p_estimated = gaussuan_mixture(μ̂, σ̂, ŵ)

histogram(x, bins=100, label="data", normed=true, alpha=0.2, xlims=(-2, 6))
plot!(x -> pdf(p, x), label="true", linewidth=4, alpha=0.8)
plot!(x -> pdf(p_estimated, x), label="estimated", linewidth=2, alpha=0.8)
for i in 1:M
    plot!(x -> pdf(Normal(μ̂[i], σ̂[i]), x), label="component $i", linewidth=2, linestyle=:dash)
end

savefig("figure/ch08/estimated.png")

log_likelihoods = [-state.log_likelihood for state in history]
plot(log_likelihoods, label="log likelihood", linewidth=2)
plot!(xscale=:log10, yscale=:log10)
xlabel!("iteration")
ylabel!("minus log likelihood")

savefig("figure/ch08/log_likelihood.png")


