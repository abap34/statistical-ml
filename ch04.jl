# 4-3.
# x_i ~^iid N(μ, σ²) として μ, σ² を最尤推定せよ
using Random
using Plots
using Distributions

N = 10000

Random.seed!(34)

μ = 3.4
σ² = 1.0

x = rand(Normal(μ, sqrt(σ²)), N)

μ̂ = sum(x) / N
σ̂² = sum((x .- μ̂).^2) / N

println("μ_ml = ", μ_ml)
println("σ²_ml = ", σ²_ml)

histogram(x, bins=100, label="data", normed=true, alpha=0.5)
plot!(x -> pdf(Normal(μ, sqrt(σ²)), x), label="true", linewidth=2)
plot!(x -> pdf(Normal(μ̂, sqrt(σ̂²)), x), label="ML", linewidth=2)

savefig("figure/ch04.png")

