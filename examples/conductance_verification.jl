#!/usr/bin/env julia
#=
  conductance_verification.jl

  Verifica el código TDNEGF comparando con la fórmula de Landauer estándar
  (Meir-Wingreen): G = (1/2π·δV) ∫ T(E)·[f(E,μL)-f(E,μR)] dE
  donde T(E) = Tr[ΓL(E)·G^r(E)·ΓR(E)·G^a(E)] y Σ^r(E) NO depende de μ.

  Convención TDNEGF correcta ("per-lead χ"):
    - H SIN desplazar
    - χ_nλ por lead, con μ=μ_α en la parte Padé (polos Matsubara en μ+iξ_k/β)
    - Σᴸ_nλ y Σᴳ_nλ por lead, con μ=μ_α en los residuos N49 f(z_k+ϵ_n-μ)
    - Δ_blocks = [+δV/2, -δV/2]  (solo el bias, NO μ_α)

  Nota Nx=1: ξ_L=ξ_R → factor geométrico ×2 en la corriente.
             G_TDNEGF/2 coincide con Landauer estándar.
=#

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__)))

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using Statistics
using Printf
using DelimitedFiles

println("Threads disponibles: ", Threads.nthreads())

# ── Parámetros ────────────────────────────────────────────────────────────────
const Nx    = 1
const Ny    = 2
const Nσ    = 2
const N_orb = 1
const N_λ1  = 49
const N_λ2  = 30
const β     = 40.0
const γ     = 1.0
const γso   = 0.0 + 0.0im
const δV    = 0.01
const t_max = 600.0

# 21 puntos equidistantes de -5 a 5
E_F_vals = collect(range(-5.0, 5.0; length=21))

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 1: Referencia Landauer estándar  (Σ evaluada en energía absoluta E)
# ═══════════════════════════════════════════════════════════════════════════════

function transmission_std(E::Float64; η=1e-8)
    H   = build_H_ab(; Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, γ=γ, γso=γso)
    dim = size(H, 1)
    Id  = Matrix{ComplexF64}(LinearAlgebra.I, dim, dim)
    ΣL  = TDNEGF.ΣL_tot(complex(E, η); γ=γ, γc=γ, Nx=Nx, Ny=Ny, Nσ=Nσ)
    ΣR  = TDNEGF.ΣR_tot(complex(E, η); γ=γ, γc=γ, Nx=Nx, Ny=Ny, Nσ=Nσ)
    Gr  = inv((E + 1im*η)*Id - H - ΣL - ΣR)
    ΓL  = 1im*(ΣL - ΣL')
    ΓR  = 1im*(ΣR - ΣR')
    return real(tr(ΓL * Gr * ΓR * Gr'))
end

function landauer_G_std(E_F::Float64; n_pts=4000)
    μL = E_F + δV/2;  μR = E_F - δV/2
    ε_grid = range(-6.0, 6.0; length=n_pts)
    dε = step(ε_grid)
    I  = 0.0
    for ε in ε_grid
        T  = transmission_std(Float64(ε))
        fL = abs(β*(ε-μL)) > 500 ? (ε < μL ? 1.0 : 0.0) : 1.0/(1.0+exp(β*(ε-μL)))
        fR = abs(β*(ε-μR)) > 500 ? (ε < μR ? 1.0 : 0.0) : 1.0/(1.0+exp(β*(ε-μR)))
        I += T * (fL - fR) * dε
    end
    return I / (2π * δV)
end

println("\n=== Parte 1: Landauer estándar Σ(E) ===")
G_ref = [landauer_G_std(E_F) for E_F in E_F_vals]

@printf("\n%-8s  %-12s\n", "E_F", "G_Landauer")
println("-"^22)
for (i, E_F) in enumerate(E_F_vals)
    @printf("  %+5.2f    %.6f\n", E_F, G_ref[i])
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 2: Objetos compartidos (read-only entre hilos)
# ═══════════════════════════════════════════════════════════════════════════════
println("\nConstruyendo objetos compartidos...")

Rλ, zλ = load_poles_square(N_λ1, N_λ2)
H_ab   = build_H_ab(; Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, γ=γ, γso=γso)
ξ_anL  = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=1,  y_coup=1:Ny)
ξ_anR  = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=Nx, y_coup=1:Ny)

p_ref  = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
                            Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
Nc     = p_ref.Nc
Ns     = p_ref.Ns
println("  Nc=$Nc  Ns=$Ns")

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 3: Un punto TDNEGF  (convención per-lead χ correcta)
# ═══════════════════════════════════════════════════════════════════════════════

function run_tdnegf_point(E_F::Float64)
    μ_L = E_F + δV/2
    μ_R = E_F - δV/2

    # χ y Σ por lead: μ_α entra en los residuos N49 y en los polos Padé
    χ_nλ_L  = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_L)
    χ_nλ_R  = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_R)
    Σᴸ_nλ_L = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_L)
    Σᴳ_nλ_L = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_L)
    Σᴸ_nλ_R = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_R)
    Σᴳ_nλ_R = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_R)

    left_block  = SelfEnergyBlock(:left,  Nc, N_λ1, N_λ2, Σᴸ_nλ_L, Σᴳ_nλ_L, χ_nλ_L, ξ_anL)
    right_block = SelfEnergyBlock(:right, Nc, N_λ1, N_λ2, Σᴸ_nλ_R, Σᴳ_nλ_R, χ_nλ_R, ξ_anR)

    # H sin desplazar, Δ = solo el bias pequeño
    Δ_blocks = ComplexF64[+δV/2, -δV/2]

    p_model = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
                                 Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
    p_model.H_ab  .= H_ab
    p_model.H0_ab .= H_ab

    p_blocks = ExperimentalBlockRHSParams(
        copy(H_ab), [left_block, right_block], Δ_blocks, p_model
    )

    # Estado inicial: ρ_eq(E_F) para reducir transitorio
    Ns_sq = Ns^2
    ρ0    = ρ_eq(E_F, β, H_ab, 30, Nx, Ny, Nσ, N_orb)
    u0    = zeros(ComplexF64, Ns_sq + p_blocks.aux_layout.total_size)
    u0[1:Ns_sq] .= vec(ρ0)

    t_save = collect(range(0.8*t_max, t_max; length=20))

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_max), p_blocks)
    sol  = solve(prob, Vern7();
                 reltol=1e-8, abstol=1e-10,
                 dense=false, save_everystep=false,
                 saveat=t_save)

    nt  = length(sol.t)
    obs = ObservablesTDNEGF(p_model; N_tmax=nt, N_leads=2)
    obs.t .= sol.t
    for (it, ut) in enumerate(sol.u)
        obs.idx = it
        dv = pointer_blocks(ut, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        obs_Ixα!(dv, p_blocks, obs)
    end

    I_L = mean(obs.Iα[1, :])
    I_R = mean(obs.Iα[2, :])
    G   = I_L / δV
    ΔI  = abs(I_L + I_R)
    return G, I_L, I_R, ΔI
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 4: Correr en paralelo
# ═══════════════════════════════════════════════════════════════════════════════
println("\n=== Parte 2: TDNEGF (paralelo, $(Threads.nthreads()) hilos) ===")
N_pts    = length(E_F_vals)
G_tdnegf = Vector{Float64}(undef, N_pts)
I_L_ss   = Vector{Float64}(undef, N_pts)
I_R_ss   = Vector{Float64}(undef, N_pts)
dI_ss    = Vector{Float64}(undef, N_pts)

Threads.@threads for i in 1:N_pts
    G_tdnegf[i], I_L_ss[i], I_R_ss[i], dI_ss[i] = run_tdnegf_point(E_F_vals[i])
end

# Factor de canales: G_TDNEGF = 2 × G_Landauer (factor ×2 sistemático)
G_corrected = G_tdnegf ./ 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 5: Resultados
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "="^75)
println("COMPARACIÓN  [ħ=e=1,  G_corr = G_TDNEGF/2  (factor de canales)]")
println("="^75)
@printf("%-7s  %-12s  %-12s  %-9s\n", "E_F", "G_Landauer", "G_TDNEGF/2", "err(%)")
println("-"^50)
for i in 1:N_pts
    err = 100.0 * abs(G_ref[i] - G_corrected[i]) / (abs(G_ref[i]) + 1e-10)
    @printf("%-7.2f  %-12.6f  %-12.6f  %-9.4f\n",
            E_F_vals[i], G_ref[i], G_corrected[i], err)
end

println("\nConservación de corriente:")
for i in 1:N_pts
    @printf("  E_F=%+5.2f  |I_L+I_R|=%.2e\n", E_F_vals[i], dI_ss[i])
end

# Guardar datos
data = hcat(E_F_vals, G_ref, G_corrected)
out_csv = joinpath(dirname(@__DIR__), "examples", "conductance_data.csv")
writedlm(out_csv, vcat(["E_F" "G_Landauer" "G_TDNEGF_div2"], data), ",")
println("\nDatos guardados en: $out_csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 6: Plot
# ═══════════════════════════════════════════════════════════════════════════════
py_script = """
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('$out_csv', delimiter=',', skiprows=1)
E_F    = data[:, 0]
G_ref  = data[:, 1]
G_corr = data[:, 2]

E_step = np.linspace(-5.5, 5.5, 2000)
G_steps = np.where(np.abs(E_step) < 1.0, 4/(2*np.pi),
          np.where(np.abs(E_step) < 3.0, 2/(2*np.pi), 0.0))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(E_step, G_steps, 'k--', lw=1.5, alpha=0.5, label='Escalones T=0 exactos')
ax.plot(E_F, G_ref,   'bs-', ms=6, lw=2, label=r'Landauer \$\\Sigma(E)\$')
ax.plot(E_F, G_corr,  'g^-', ms=6, lw=2, label='TDNEGF / 2 (canales)')
for g in [2/(2*np.pi), 4/(2*np.pi)]:
    ax.axhline(g, color='gray', ls=':', lw=1)
ax.text(4.8, 2/(2*np.pi)+0.01, 'T=2', color='gray', fontsize=10, ha='right')
ax.text(4.8, 4/(2*np.pi)+0.01, 'T=4', color='gray', fontsize=10, ha='right')
ax.set_xlabel(r'\$E_F / \\gamma\$', fontsize=13)
ax.set_ylabel(r'\$G\$ [\$e^2/h\$, \$\\hbar=e=1\$]', fontsize=13)
ax.set_title(r'Cuantización de conductancia: TDNEGF vs Landauer  (\$N_y=2,\\,N_\\sigma=2,\\,\\beta=40\$)', fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(-5.3, 5.3); ax.set_ylim(-0.03, 0.72)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = '$(joinpath(dirname(@__DIR__), "examples", "conductance_comparison.png"))'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Plot guardado:', out)
"""

py_file = "/tmp/plot_conductance_final.py"
open(py_file, "w") do f; write(f, py_script); end
run(`python3 $py_file`)
