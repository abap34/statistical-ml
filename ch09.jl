# 9-2.
# 平均 0.5, 分散 1 の正規分布から n個の訓練標本を生成して、
# 最尤推定量と最大事後確率推定量を求めて比較せよ
# (事前確率として 平均 0, 分散 γ の正規分布を仮定する)

using Random
using Distributions
using Plots

Random.seed!(34)

const μ = 0.5
const σ = 1.0
const xlims = (-5, 5)

function generate_data(n)
    return rand(Normal(μ, σ), n)
end


# 最尤推定量
function mle(x)
    μ̂ = sum(x) / length(x)
    σ̂² = sum((x .- μ̂) .^ 2) / length(x)
    return μ̂, σ̂²
end


# 最大事後確率推定量 (事前確率として 平均 0, 分散 γ の正規分布を仮定)
function map(x, γ)
    μ̂ = sum(x) / (length(x) + inv(γ))
    σ̂² = sum((x .- μ̂) .^ 2) / (length(x) + 2 * inv(γ))
    return μ̂, σ̂²
end

function experiment(n, γ)
    x = generate_data(n)
    μ̂_mle, σ̂²_mle = mle(x)
    μ̂_map, σ̂²_map = map(x, γ)
    return μ̂_mle, σ̂²_mle, μ̂_map, σ̂²_map
end

function plot_result(x, μ̂_mle, σ̂²_mle, μ̂_map, σ̂²_map)
    histogram(x, bins=min(100, length(x)), label="data", normed=true, alpha=0.5)
    plot!(x -> pdf(Normal(μ̂_mle, sqrt(σ̂²_mle)), x), label="MLE", linewidth=2, xlims=xlims)
    plot!(x -> pdf(Normal(μ̂_map, sqrt(σ̂²_map)), x), label="MAP", linewidth=2, xlims=xlims)
end

function main()
    N_values = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    γ = 0.1

    for N in N_values
        μ̂_mle, σ̂²_mle, μ̂_map, σ̂²_map = experiment(N, γ)
        plot_result(generate_data(N), μ̂_mle, σ̂²_mle, μ̂_map, σ̂²_map)
        title!("N = $N")
        savefig("figure/ch09/N_$(N).png")
    end

end

main()

