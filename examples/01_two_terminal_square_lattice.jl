#!/usr/bin/env julia

#=
Example: 2D square lattice device coupled to square-lattice leads

This example now uses the heterogeneous block-based TDNEGF backend as the
primary workflow:
  * Build the same square-lattice Hamiltonian H_ab for a finite Nx × Ny system.
  * Build lead ingredients (Σᴸ/Σᴳ/χ and ξ) as before.
  * Assemble explicit SelfEnergyBlock objects (left/right).
  * Construct ExperimentalBlockRHSParams and solve eom_tdnegf_blocks!.
  * Compute observables directly from pointer_blocks(...) without converting
    states back to the legacy rectangular auxiliary layout.

The legacy rectangular path remains available in TDNEGF.jl for compatibility,
but this script intentionally documents the block path first.
=#
using Pkg
Pkg.activate(joinpath(dirname((@__DIR__))))

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using BenchmarkTools

println("Number of threads with JUlIA_NUM used in operations is : ", Threads.nthreads())
println("Number of threads with BLAS used in operations is : ", BLAS.get_num_threads())

#### Define model and initial parameters
function init_params_blocks(;Nx::Int=50, Ny::Int=2, Nσ::Int=2, N_orb::Int=1,
                            γ::Float64=1.0, γso=0.5 + 0.0im, Nα::Int=2,
                            N_λ1::Int=49, N_λ2::Int=20, β::Float64=33.0)
    Rλ, zλ = load_poles_square(N_λ1, N_λ2)

    # Keep a light ModelParams object for lattice geometry and observable indexing.
    p_model = ModelParamsTDNEGF(Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb,
                                Nα = Nα, N_λ1 = N_λ1, N_λ2 = N_λ2)

    H_ab  = build_H_ab(; Nx = p_model.Nx, Ny = p_model.Ny, Nσ = p_model.Nσ,
                        N_orb = p_model.N_orb, γ = γ, γso = γso)
    Σᴸ_nλ = build_Σᴸ_nλ(Rλ, zλ, p_model.Ny, p_model.Nσ, p_model.N_orb,
                        p_model.N_λ1, p_model.N_λ2; β = β, γ = 1.0)
    Σᴳ_nλ = build_Σᴳ_nλ(Rλ, zλ, p_model.Ny, p_model.Nσ, p_model.N_orb,
                        p_model.N_λ1, p_model.N_λ2; β = β, γ = 1.0)
    χ_nλ  = build_χ_nλ(zλ, p_model.Ny, p_model.Nσ, p_model.N_orb,
                       p_model.N_λ1, p_model.N_λ2; β = β, γ = 1.0)

    ξ_anR = build_ξ_an(p_model.Nx, p_model.Ny, p_model.Nσ, p_model.N_orb;
                       xcol = p_model.Nx, y_coup = 1:p_model.Ny)
    ξ_anL = build_ξ_an(p_model.Nx, p_model.Ny, p_model.Nσ, p_model.N_orb;
                       xcol = 1, y_coup = 1:p_model.Ny)

    left_block = SelfEnergyBlock(:left, p_model.Nc, p_model.N_λ1, p_model.N_λ2,
                                 Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anL, 0.5 + 0.0im)
    right_block = SelfEnergyBlock(:right, p_model.Nc, p_model.N_λ1, p_model.N_λ2,
                                  Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anR, -0.5 + 0.0im)
    blocks = [left_block, right_block]

    p_model.H_ab .= H_ab
    p_model.H0_ab .= H_ab

    # Block RHS params are the main backend object for solve + current observables.
    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, p_model)

    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)
    return p_model, p_blocks, u0
end

p_model, p_blocks, u0 = init_params_blocks()
println("The initial parameters have been set")

#### Define and solve the ODE problem
prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, 100.0), p_blocks)
println("The integrator has been set")

@time sol = solve(prob, Vern7(), adaptive=true, dense=false, reltol=1e-6, abstol=1e-8)
println("Solution was obtained")

#### Allocate and compute observables directly from block pointers
obs = ObservablesTDNEGF(p_model; N_tmax=length(sol.t), N_leads=length(p_blocks.blocks))
obs.t = sol.t
for (it, ut) in enumerate(sol.u)
    obs.idx = it
    dv = pointer_blocks(ut, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
    obs_n_i!(dv, p_model, obs)      ### Charge density
    obs_σ_i!(dv, p_model, obs)      ### Spin density
    obs_Ixα!(dv, p_blocks, obs)     ### Spin and charge current from block Π_abα
end

#### Save results
using JLD2
@save "./examples/data/two_terminal_square_lattice_observables.jl2" obs
println("The results has been saved")

#### Plot results
using PyPlot
plt.rc("axes", linewidth=1)
plt.rc("text", usetex=true)
fs = 25

fig, axs = plt.subplots(1, 1)
sites = 1:2:12
for site in sites
    axs.plot(obs.t, obs.n_i[site, :], label="site=$(site)")
end
axs.set_ylabel(raw"$\langle\mathrm{\hat{n}_i}\rangle$", fontsize=fs)
axs.set_xlabel(raw"$\mathrm{Time\ (\hbar/\gamma)}$", fontsize=fs)
axs.tick_params(axis="both", which="both", labelsize=fs, direction="in", length=6, width=1)
axs.ticklabel_format(axis="y", style="sci", scilimits=(-1, 2), useMathText=true)
axs.yaxis.offsetText.set_fontsize(fs)
plt.legend(frameon=false, fontsize=fs - 10, loc=(0.7, 0.1))
plt.tight_layout()
plt.show()
