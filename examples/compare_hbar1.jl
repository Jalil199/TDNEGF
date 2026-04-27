using DelimitedFiles, JLD2

# Legacy hbar=1
curr_raw = vec(readdlm("examples/data/legacy_hbar1_curr.txt", Float64))
n_t = length(curr_raw) ÷ 2
I_L_h1 = curr_raw[1:2:end]

cden_raw = vec(readdlm("examples/data/legacy_hbar1_cden.txt", Float64))
charge_h1 = [sum(cden_raw[(i-1)*40+1:i*40]) for i in 1:n_t]

# Codigo nuevo N49
d = load("examples/data/benchmark_legacy_n49.jld2")
t_new      = d["t"]
I_L_raw    = d["I_L"]
I_L_new    = ndims(I_L_raw) == 1 ? I_L_raw : vec(I_L_raw[1, :])
charge_new = d["total_charge"]
println("shapes: t_new=$(size(t_new))  I_L=$(size(I_L_raw))  charge=$(size(charge_new))")

println("=== Tr(rho) comparacion ===")
for t in [10, 25, 50]
    idx_new = findfirst(t_new .>= Float64(t))
    println("t=$t:  legacy_h1=$(round(charge_h1[t],digits=4))  nuevo=$(round(charge_new[idx_new],digits=4))")
end

println()
println("=== I_L comparacion ===")
for t in [10, 25, 50]
    idx_new = findfirst(t_new .>= Float64(t))
    println("t=$t:  legacy_h1=$(round(I_L_h1[t],digits=6))  nuevo=$(round(I_L_new[idx_new],digits=6))")
end
