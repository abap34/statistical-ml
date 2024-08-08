# 6.2 線形判別分析で MNIST を分類せよ

# 以下の仮定を置く:
# - カテゴリ c のパターンは N(μ_c, Σ) にしたがう (Σ が全クラスで共通)
using MLDatasets
using Statistics
using LinearAlgebra
using ProgressMeter
using Plots

X, y = MNIST(split=:train)[:]
X = reshape(X, 28*28, :)
X_test, y_test = MNIST(split=:test)[:]
X_test = reshape(X_test, 28*28, :)


C = 0:9

N = size(X, 2)
N_test = size(X_test, 2)

@assert size(X) == (28*28, N)
@assert size(y) == (N,)

@assert size(X_test) == (28*28, N_test)
@assert size(y_test) == (N_test,)


# カテゴリごとの訓練標本
X_grouped = Dict(c => X[:, y .== c] for c in C)
N_c = Dict(c => size(X_grouped[c], 2) for c in C)

@assert sum(N_c[c] for c in C) == N

# カテゴリごとの標本平均 (最尤推定量)
μ = Dict(c => mean(X_grouped[c], dims=2) for c in C)

@assert size(μ[0]) == (28*28, 1)

# 共通の分散共分散行列を最尤推定する.
# 共通の分散共分散行列は、各カテゴリの分散共分散行列の標本数による重み付き平均 
Σ = sum(cov(X_grouped[c], dims=2) * N_c[c] for c in C) / N

# 逆行列が存在しない場合、正定値性を保証するための小さな値を加える
if !(isposdef(Σ))
    Σ = Σ + 0.01 * I
end

Σ⁻¹ = inv(Σ)

# カテゴリ c の対数事後確率 (定数項は無視)
function posterior(x, c)::Float64
    μ_c = μ[c]
    logp = x' * Σ⁻¹ * μ_c .- 0.5 * μ_c' * Σ⁻¹ * μ_c
    return logp[1]
end
    
# 最大事後確率則による分類
function predict(x)
    return argmax(posterior(x, c) for c in C) - 1
end

X_test, y_test = MNIST(split=:test)[:]
X_test = reshape(X_test, 28*28, :)

N_test = size(X_test, 2)
y_pred = similar(y_test)

@showprogress for i in 1:N_test
    x = X_test[:, i]
    y_pred[i] = predict(x)
end

accuracy = sum(y_pred .== y_test) / N_test

println("Accuracy: $accuracy")

# 混同行列
confusion_matrix = zeros(Int, 10, 10)

for i in 1:N_test
    confusion_matrix[y_test[i] + 1, y_pred[i] + 1] += 1
end

# 数字も表示
heatmap(confusion_matrix, c=:blues, xlabel="Predicted", ylabel="True", title="Confusion Matrix", 
        xticks=(1:10, 0:9), yticks=(1:10, 0:9), aspect_ratio=1, fmt=:d)

savefig("figure/ch06.png")