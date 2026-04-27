using DelimitedFiles, JLD2, PyPlot

# ── Legacy hbar=1 ──────────────────────────────────────────────
curr_raw = vec(readdlm("examples/data/legacy_hbar1_curr.txt", Float64))
n_t      = length(curr_raw) ÷ 2
I_L_h1   = curr_raw[1:2:end]
t_leg    = collect(1.0:1.0:n_t)

cden_raw  = vec(readdlm("examples/data/legacy_hbar1_cden.txt", Float64))
charge_h1 = [sum(cden_raw[(i-1)*40+1:i*40]) for i in 1:n_t]

# ── Legacy hbar=0.658 (referencia) ─────────────────────────────
curr_ref = vec(readdlm("examples/data/legacy_bench_curr.txt", Float64))
n_ref    = length(curr_ref) ÷ 2
I_L_ref  = curr_ref[1:2:end]
t_ref    = collect(1.0:1.0:n_ref)

cden_ref    = vec(readdlm("examples/data/legacy_bench_cden.txt", Float64))
charge_ref  = [sum(cden_ref[(i-1)*40+1:i*40]) for i in 1:n_ref]

# ── Codigo nuevo N49 ───────────────────────────────────────────
d          = load("examples/data/benchmark_legacy_n49.jld2")
t_new      = d["t"]
I_L_new    = d["I_L"]
charge_new = d["total_charge"]

# ── Plot ───────────────────────────────────────────────────────
fig, axes = subplots(1, 2, figsize=(12, 4))

ax1 = axes[1]
ax1.plot(t_ref[1:50],    charge_ref[1:50],  "k-",  lw=2,   label="legacy  ħ=0.658")
ax1.plot(t_leg,          charge_h1,         "r--", lw=2,   label="legacy  ħ=1")
ax1.plot(t_new,          charge_new,        "b:",  lw=2,   label="nuevo N49")
ax1.set_xlabel("t")
ax1.set_ylabel("Tr(ρ)")
ax1.set_title("Carga total")
ax1.legend()
ax1.set_xlim(0, 50)
ax1.grid(true, alpha=0.3)

ax2 = axes[2]
ax2.plot(t_ref[1:50],    I_L_ref[1:50],     "k-",  lw=2,   label="legacy  ħ=0.658")
ax2.plot(t_leg,          I_L_h1,            "r--", lw=2,   label="legacy  ħ=1")
ax2.plot(t_new,          I_L_new,           "b:",  lw=2,   label="nuevo N49")
ax2.set_xlabel("t")
ax2.set_ylabel("I_L")
ax2.set_title("Corriente izquierda")
ax2.legend()
ax2.set_xlim(0, 50)
ax2.grid(true, alpha=0.3)

tight_layout()
savefig("examples/hbar1_comparison.png", dpi=150)
println("Guardado: examples/hbar1_comparison.png")
