# 6.2 線形判別分析で MNIST を分類せよ
using MLDatasets
using Statistics
using LinearAlgebra
using PrettyTables
using OffsetArrays
using Plots

X, y = MNIST(split=:train)[:]
X = reshape(X, 28 * 28, :)

X_test, y_test = MNIST(split=:test)[:]
X_test = reshape(X_test, 28 * 28, :)

C = 0:9
d = 28 * 28

N = size(X, 2)
N_test = size(X_test, 2)

@assert size(X) == (d, N)
@assert size(y) == (N,)

@assert size(X_test) == (28 * 28, N_test)
@assert size(y_test) == (N_test,)


# 以下の仮定を置く:
# - カテゴリ c のパターンは N(μ_c, Σ) にしたがう (Σ が全クラスで共通)


# カテゴリごとの訓練標本, 標本数, 標本平均 (= μ_c の最尤推定量) 
X_c = OffsetArray{Matrix{Float64}}(undef, 0:9)
N_c = OffsetArray{Int}(undef, 0:9)
μ̂_c = OffsetArray{Vector{Float64}}(undef, 0:9)

for c in C
    X_c[c] = X[:, y.==c]
    N_c[c] = size(X_c[c], 2)
    μ̂_c[c] = mean(X_c[c], dims=2) |> vec
end

for c in C
    # 数字の向きが左が上になってるので右に90度回転させる
    img = μ̂_c[c] |> (x -> reshape(x, 28, 28)) |> transpose |> (x -> reverse(x, dims=1))
    heatmap(img, aspect_ratio=1, title="c = $c", color=:grays)
    savefig("figure/ch06/mnist-μ̂_c-$c.png")
end

@assert sum(N_c) == N
@assert all(size(μ̂_c[c]) == (d,) for c in C)

# 共通の分散共分散行列を最尤推定する.
# 共通の分散共分散行列は、各カテゴリの分散共分散行列の標本数による重み付き平均 
Σ̂ = sum(cov(X_c[c], dims=2) * N_c[c] for c in C) / N

@assert size(Σ̂) == (d, d)

# 正定値性を確認し、正定値でない場合は微小値を足して正定値にする
if !(isposdef(Σ̂))
    Σ̂ += 1e-6 * I
end

@assert isposdef(Σ̂)

Σ̂⁻¹ = inv(Σ̂)
μ̂ = hcat(μ̂_c...)

# 事後確率の第二項と第三項を前計算しておく. (カテゴリのみに依り、 x に依らないので)
# 事後確率の第二項
p2 = [-0.5 * μ̂_c[c]' * Σ̂⁻¹ * μ̂_c[c] for c in C]
# 事後確率の第三項
p3 = log.(OffsetArrays.no_offset_view(N_c))
p_c = p2 .+ p3

# カテゴリ c の事後確率 (パターンにもカテゴリにも依らない項は省略)
function posterior_batch(X::Matrix)::Matrix{Float64}
    p1 = X' * Σ̂⁻¹ * μ̂
    return p1 .+ p_c'
end

# 最大事後確率則による分類
ŷ = argmax(posterior_batch(X), dims=2) .|> (x -> x[2] - 1)
ŷ_test = argmax(posterior_batch(X_test), dims=2) .|> (x -> x[2] - 1)

# 訓練誤差
accuracy = sum(ŷ .== y) / N
println("Train Accuracy: $accuracy")

# テスト誤差
accuracy_test = sum(ŷ_test .== y_test) / N_test
println("Test Accuracy: $accuracy_test")

function viz_confusion_matrix(confusion_matrix::Matrix{Int})
    confusion_matrix = hcat(0:9, confusion_matrix)
    highliter = Highlighter(
        (data, i, j) -> (j > 0) && (i + 1 == j),
        crayon"green"
    )
    header = ["true / pred", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    pretty_table(confusion_matrix, header=header, highlighters=(highliter,))
end

# 混同行列
confusion_matrix = zeros(Int, 10, 10)
for i in 1:N_test
    confusion_matrix[y_test[i]+1, ŷ_test[i]+1] += 1
end
viz_confusion_matrix(confusion_matrix)