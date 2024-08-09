# 7-2.
# 以下の 3つのガウスモデルを考える.
# f(Σ) = N(μ, Σ) として、
# (A) { f(Σ) | Σ ∈ {正定値対称行列} }
# (B) { f(Σ) | Σ ∈ {対角行列} }
# (C) { f(Σ) | Σ ∈ {対角成分が等しい対角行列} }
# 適当な2次元分布から訓練標本を生成して、　AIC によりモデル選択を行え
# 訓練標本の個数と真の分布を変化させて、モデル選択の結果がどのように変化するか確認せよ

using Random
using Distributions
using LinearAlgebra
using Statistics
using PrettyTables
using Plots
using DataFrames

Random.seed!(34)

const d = 2

function experiment(p::Sampleable, N::Int)
    X = rand(p, N)

    @assert size(X) == (d, N)

    # モデル A のパラメータを最尤推定
    μ̂_A = mean(X, dims=2) |> vec
    Σ̂_A = cov(X, dims=2)

    # 自由度
    t_A = d * (d + 1) / 2

    @assert size(μ̂_A) == (d,)
    @assert size(Σ̂_A) == (d, d)


    # モデル B のパラメータを最尤推定
    μ̂_B = mean(X, dims=2) |> vec
    σ̂ʲ = similar(μ̂_B)
    for j in 1:d
        σ̂ʲ[j] = sqrt(sum((X[j, i] - μ̂_B[j])^2 for i in 1:N) / N)
    end

    Σ̂_B = diagm(σ̂ʲ .^ 2)

    # 自由度
    t_B = d


    # モデル C のパラメータを最尤推定
    μ̂_C = mean(X, dims=2) |> vec
    σ̂² = sum(σ̂ʲ .^ 2) / d
    Σ̂_C = diagm(fill(σ̂², d))

    # 自由度
    t_C = 1


    # 赤池情報量基準を計算

    logl_A = loglikelihood(MvNormal(μ̂_A, Σ̂_A), X)
    logl_B = loglikelihood(MvNormal(μ̂_B, Σ̂_B), X)
    logl_C = loglikelihood(MvNormal(μ̂_C, Σ̂_C), X)


    AIC_A = -logl_A + t_A
    AIC_B = -logl_B + t_B
    AIC_C = -logl_C + t_C

    return DataFrame(
        Model=["A", "B", "C"],
        LogLikelihood=[logl_A, logl_B, logl_C],
        NumberOfParameters=[t_A, t_B, t_C],
        AIC=[AIC_A, AIC_B, AIC_C]
    )
end

function viz(df::DataFrame)
    # AIC 最大の行をハイライト
    max_idx = argmax(df.AIC)

    highligher = Highlighter(
        (data, i, j) -> (i == max_idx),
        crayon"yellow"
    )

    PrettyTables.pretty_table(df, highlighters=(highligher,))
end

function main()
    # 適当な 2 次元正規分布 
    μ = [3.0, 4.0]
    Σ = [3.0 1.0;
        1.0 2.0]

    p = MvNormal(μ, Σ)
    N = 100

    result = experiment(p, N)
    viz(result)

    N_values = [5, 10, 50, 100, 500]

    AIC_A = Float64[]
    AIC_B = Float64[]
    AIC_C = Float64[]

    for N in N_values
        result = experiment(p, N)
        push!(AIC_A, result.AIC[1])
        push!(AIC_B, result.AIC[2])
        push!(AIC_C, result.AIC[3])
    end

    plot(N_values, AIC_A, label="A", linewidth=2)
    plot!(N_values, AIC_B, label="B", linewidth=2)
    plot!(N_values, AIC_C, label="C", linewidth=2)
    plot!(xscale=:log10, yscale=:log10)
    xlabel!("Number of Samples")
    ylabel!("AIC")
    savefig("figure/ch07/aic_norm.png")

    # 2 次元一様分布 [-1, 1] x [-1, 1]
    p = product_distribution([Uniform(-1, 1), Uniform(-1, 1)])
    N = 100

    result = experiment(p, N)
    viz(result)

    AIC_A = Float64[]
    AIC_B = Float64[]
    AIC_C = Float64[]

    for N in N_values
        result = experiment(p, N)
        push!(AIC_A, result.AIC[1])
        push!(AIC_B, result.AIC[2])
        push!(AIC_C, result.AIC[3])
    end

    plot(N_values, AIC_A, label="A", linewidth=2)
    plot!(N_values, AIC_B, label="B", linewidth=2)
    plot!(N_values, AIC_C, label="C", linewidth=2)
    plot!(xscale=:log10, yscale=:log10)
    xlabel!("Number of Samples")
    ylabel!("AIC")
    savefig("figure/ch07/aic_unif.png")
end

main()












