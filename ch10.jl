# 10.3
# ラプラス分布に従う確率変数を逆関数サンプリング法を用いて生成せよ.
# また、 逆関数サンプリング法によって生成した標本を使って g(x) = x^2  の　ラプラス分布 p(x) = 1/2 exp(-|x|) に関する期待値をモンテカルロ積分により計算せよ.

using Random
using Distributions
using Plots

Random.seed!(34)

N = 10^4

# ラプラス分布の累積分布関数
cdf(x) = (1 + sign(x) * (1 - exp(-abs(x)))) / 2

# ラプラス分布の累積分布関数の逆関数
inv_cdf(y) = -sign(y - 0.5) * log(1 - 2 * abs(y - 0.5))

# 一様分布からラプラス分布に従う確率変数を生成
x = inv_cdf.(rand(N))

histogram(x, bins=100, label="Inverse Transform Sampling", normed=true, alpha=0.5)
histogram!(rand(Laplace(), N), bins=100, label="Laplace", normed=true, alpha=0.5)
savefig("figure/ch10/laplace.png")


# モンテカルロ積分
function montecalro_integration(f, N)
    x = inv_cdf.(rand(N))
    return mean(f, x)
end

f(x) = x^2

N_values = [10^i for i in 1:6]

plot(
    N_values,
    [montecalro_integration(f, N) for N in N_values],
    xscale=:log10,
    yscale=:log10,
    label="Monte Carlo Integration",
    xlabel="N",
    ylabel="Estimate",
    title="Monte Carlo Integration of x^2",
)

hline!([2], label="True Value", linewidth=2, linestyle=:dash)

savefig("figure/ch10/montecalro_integration.png")

