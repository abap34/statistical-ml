# 5-2.
# 一次元一様分布に独立に従う \{ x_i \}_{i=1}^N を生成し、その平均 \bar{x} を計算することで
# 中心極限定理を確認せよ

using Random
using Plots
using Distributions

N_values = [1, 2, 5, 10, 50, 100, 500, 1000, 5000]  
N_trial = 10000  

Random.seed!(34)

function sample_mean(N)
    x = rand(Uniform(0, 1), N)
    return mean(x)
end


anim = @animate for N in N_values
    sample_means = [sample_mean(N) for _ in 1:N_trial]
    
    histogram(sample_means, bins=100, label="Data", normed=true, alpha=0.5, 
              title="N = $N", xlabel="Sample Mean", ylabel="Frequency", xlims=(0, 1))
    
    plot!(x -> pdf(Normal(0.5, sqrt(1/(12 * N))), x), label="Theory", linewidth=2, xlims=(0, 1))
end 

gif(anim, "figure/ch05.gif", fps=2)