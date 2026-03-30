#!/usr/bin/env julia

#=
Example 4: Open 1D two-channel benchmark (block TDNEGF backend)

Baseline electronic benchmark for later workflows with light / HF / classical
moments. This script runs two cases with the same code path:
  A) no-gap central region
  B) central-gap region (orbital-dependent onsite energies only in the device)

Local basis per site:
  (c↑, c↓, v↑, v↓)
with Nx × Ny = Nx × 1 sites.
=#

using Pkg
Pkg.activate(joinpath(dirname((@__DIR__))))

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using Statistics
using JLD2
using PyPlot

"Return global basis index for (x,y,orb,spin)."
@inline function idx_state(x::Int, y::Int, orb::Int, σ::Int; Ny::Int, Nσ::Int, N_orb::Int)
    N_loc = Nσ * N_orb
    site = (x - 1) * Ny + y
    local = (orb - 1) * Nσ + σ
    return (site - 1) * N_loc + local
end

"Build 1D (Ny=1) two-orbital spin-diagonal central Hamiltonian."
function build_H_ab_1d_twochannel(; Nx::Int, Ny::Int=1, Nσ::Int=2, N_orb::Int=2,
                                  tc::Float64=-1.0, tv::Float64=+1.0,
                                  eps_c::Float64=0.0, eps_v::Float64=0.0)
    Ny == 1 || throw(ArgumentError("This helper is intended for Ny=1 benchmark"))
    N_orb == 2 || throw(ArgumentError("This helper assumes two orbitals (c,v)"))

    Ns = Nx * Ny * Nσ * N_orb
    H = zeros(ComplexF64, Ns, Ns)

    # onsite terms (spin-diagonal, orbital-diagonal)
    @inbounds for x in 1:Nx, σ in 1:Nσ
        ic = idx_state(x, 1, 1, σ; Ny, Nσ, N_orb)
        iv = idx_state(x, 1, 2, σ; Ny, Nσ, N_orb)
        H[ic, ic] = eps_c
        H[iv, iv] = eps_v
    end

    # nearest-neighbor hopping along x, no spin/orbital mixing
    @inbounds for x in 1:(Nx - 1), σ in 1:Nσ
        i1 = idx_state(x, 1, 1, σ; Ny, Nσ, N_orb)
        j1 = idx_state(x + 1, 1, 1, σ; Ny, Nσ, N_orb)
        H[i1, j1] = tc
        H[j1, i1] = tc

        i2 = idx_state(x, 1, 2, σ; Ny, Nσ, N_orb)
        j2 = idx_state(x + 1, 1, 2, σ; Ny, Nσ, N_orb)
        H[i2, j2] = tv
        H[j2, i2] = tv
    end

    return H
end

"Assemble per-orbital lead residues using existing square-lead helpers."
function build_twochannel_lead_coeffs(Rλ, zλ; Nσ::Int, N_λ1::Int, N_λ2::Int,
                                      β::Float64, tc::Float64, tv::Float64)
    ΣL_c = build_Σᴸ_nλ(Rλ, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=abs(tc))
    ΣG_c = build_Σᴳ_nλ(Rλ, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=abs(tc))
    χ_c  = build_χ_nλ(zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=abs(tc))

    ΣL_v = build_Σᴸ_nλ(Rλ, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=abs(tv))
    ΣG_v = build_Σᴳ_nλ(Rλ, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=abs(tv))
    χ_v  = build_χ_nλ(zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=abs(tv))

    Nλ = N_λ1 + N_λ2
    Nc = 2 * Nσ
    ΣL = zeros(ComplexF64, Nc, Nλ)
    ΣG = zeros(ComplexF64, Nc, Nλ)
    χ  = zeros(ComplexF64, Nc, Nλ)

    ΣL[1:Nσ, :] .= ΣL_c
    ΣL[Nσ+1:2Nσ, :] .= ΣL_v
    ΣG[1:Nσ, :] .= ΣG_c
    ΣG[Nσ+1:2Nσ, :] .= ΣG_v
    χ[1:Nσ, :] .= χ_c
    χ[Nσ+1:2Nσ, :] .= χ_v
    return ΣL, ΣG, χ
end

"Initializer for Example 4 (block backend, 1D two-channel benchmark)."
function init_params_blocks_1d_twochannel(; Nx::Int=10, Ny::Int=1, Nσ::Int=2, N_orb::Int=2,
                                          tc::Float64=-1.0, tv::Float64=+1.0,
                                          Δ::Float64=1.0,
                                          eps_c_lead::Float64=0.0, eps_v_lead::Float64=0.0,
                                          Vbias::Float64=0.2,
                                          t_end::Float64=80.0, dt::Float64=0.2,
                                          N_λ1::Int=49, N_λ2::Int=20,
                                          β::Float64=33.0)
    Ny == 1 || throw(ArgumentError("Example 4 is configured for Ny=1"))
    Nσ == 2 || throw(ArgumentError("Example 4 assumes spin-1/2 (Nσ=2)"))
    N_orb == 2 || throw(ArgumentError("Example 4 assumes two orbitals (N_orb=2)"))

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    ΣL_nλ, ΣG_nλ, χ_nλ = build_twochannel_lead_coeffs(Rλ, zλ; Nσ, N_λ1, N_λ2, β, tc, tv)

    return (
        Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
        tc=tc, tv=tv, Δ=Δ,
        eps_c_lead=eps_c_lead, eps_v_lead=eps_v_lead,
        Vbias=Vbias, t_end=t_end, dt=dt,
        N_λ1=N_λ1, N_λ2=N_λ2, β=β,
        ΣL_nλ=ΣL_nλ, ΣG_nλ=ΣG_nλ, χ_nλ=χ_nλ,
    )
end

"Run one benchmark case and collect observables."
function run_case_blocks_1d_twochannel(cfg; eps_c_center::Float64, eps_v_center::Float64, label::String)
    p_model = ModelParamsTDNEGF(
        Nx=cfg.Nx, Ny=cfg.Ny, Nσ=cfg.Nσ, N_orb=cfg.N_orb,
        Nα=2, N_λ1=cfg.N_λ1, N_λ2=cfg.N_λ2,
    )

    H = build_H_ab_1d_twochannel(; Nx=cfg.Nx, Ny=cfg.Ny, Nσ=cfg.Nσ, N_orb=cfg.N_orb,
                                 tc=cfg.tc, tv=cfg.tv,
                                 eps_c=eps_c_center, eps_v=eps_v_center)
    p_model.H_ab .= H
    p_model.H0_ab .= H

    ξ_anL = build_ξ_an(cfg.Nx, cfg.Ny, cfg.Nσ, cfg.N_orb; xcol=1, y_coup=1:cfg.Ny)
    ξ_anR = build_ξ_an(cfg.Nx, cfg.Ny, cfg.Nσ, cfg.N_orb; xcol=cfg.Nx, y_coup=1:cfg.Ny)

    left_block = SelfEnergyBlock(:left, p_model.Nc, cfg.N_λ1, cfg.N_λ2,
                                 cfg.ΣL_nλ, cfg.ΣG_nλ, cfg.χ_nλ, ξ_anL)
    right_block = SelfEnergyBlock(:right, p_model.Nc, cfg.N_λ1, cfg.N_λ2,
                                  cfg.ΣL_nλ, cfg.ΣG_nλ, cfg.χ_nλ, ξ_anR)
    blocks = [left_block, right_block]
    Δ_blocks = ComplexF64[+cfg.Vbias/2, -cfg.Vbias/2]

    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, Δ_blocks, p_model)
    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, cfg.t_end), p_blocks)
    sol = solve(prob, Vern7(); saveat=cfg.dt, adaptive=true, dense=false,
                reltol=1e-6, abstol=1e-8)

    obs = ObservablesTDNEGF(p_model; N_tmax=length(sol.t), N_leads=length(blocks))
    obs.t = sol.t
    for (it, ut) in enumerate(sol.u)
        obs.idx = it
        ptr = pointer_blocks(ut, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        obs_n_i!(ptr, p_model, obs)
        obs_σ_i!(ptr, p_model, obs)
        obs_Ixα!(ptr, p_blocks, obs)
    end

    return (
        label=label,
        p_model=p_model,
        p_blocks=p_blocks,
        H=H,
        obs=obs,
        eps_c_center=eps_c_center,
        eps_v_center=eps_v_center,
    )
end



"Bloch Hamiltonian H(k) for the infinite 1D two-orbital chain in basis (c↑, c↓, v↑, v↓)."
@inline function bloch_Hk_1d_twochannel(k::Float64; tc::Float64, tv::Float64,
                                       eps_c::Float64, eps_v::Float64)
    ec = eps_c + 2 * tc * cos(k)
    ev = eps_v + 2 * tv * cos(k)
    # spin-degenerate, orbital-diagonal 4x4 Bloch Hamiltonian
    return ComplexF64[
        ec 0  0  0;
        0  ec 0  0;
        0  0  ev 0;
        0  0  0  ev
    ]
end

"Compare infinite-chain Bloch dispersions for no-gap and central-gap parameter sets."
function plot_bloch_dispersion_comparison(cfg)
    kgrid = range(-π, π; length=400)
    E_nogap = Matrix{Float64}(undef, 4, length(kgrid))
    E_gap = Matrix{Float64}(undef, 4, length(kgrid))

    for (ik, k) in enumerate(kgrid)
        Hk0 = bloch_Hk_1d_twochannel(k; tc=cfg.tc, tv=cfg.tv,
                                     eps_c=cfg.eps_c_lead, eps_v=cfg.eps_v_lead)
        Hkg = bloch_Hk_1d_twochannel(k; tc=cfg.tc, tv=cfg.tv,
                                     eps_c=+cfg.Δ/2, eps_v=-cfg.Δ/2)
        E_nogap[:, ik] .= sort(real(eigvals(Hk0)))
        E_gap[:, ik] .= sort(real(eigvals(Hkg)))
    end

    fig, axs = subplots(1, 2, figsize=(11, 4), sharey=true)
    for b in 1:4
        axs[1].plot(kgrid, E_nogap[b, :], color="tab:blue", linewidth=1.2)
        axs[2].plot(kgrid, E_gap[b, :], color="tab:red", linewidth=1.2)
    end
    axs[1].set_title("Bloch bands (no-gap)")
    axs[2].set_title("Bloch bands (central-gap)")
    for ax in axs
        ax.set_xlabel(L"k")
        ax.grid(true, alpha=0.3)
    end
    axs[1].set_ylabel(L"E(k)")
    axs[2].axhline(+cfg.Δ/2, color="k", linestyle="--", linewidth=1.0)
    axs[2].axhline(-cfg.Δ/2, color="k", linestyle="--", linewidth=1.0)
    suptitle("Infinite central-region Bloch dispersion")
    tight_layout()
    return fig
end

"Plot analytic lead dispersions Ec(k), Ev(k)."
function plot_lead_dispersion(cfg)
    k = range(-π, π; length=400)
    Ec = cfg.eps_c_lead .+ 2 .* cfg.tc .* cos.(k)
    Ev = cfg.eps_v_lead .+ 2 .* cfg.tv .* cos.(k)

    fig, ax = subplots(1, 1, figsize=(6, 4))
    ax.plot(k, Ec, label=L"E_c(k)")
    ax.plot(k, Ev, label=L"E_v(k)")
    ax.set_xlabel(L"k")
    ax.set_ylabel(L"E")
    ax.set_title("Lead dispersions")
    ax.legend(frameon=false)
    ax.grid(true, alpha=0.3)
    tight_layout()
    return fig
end

"Plot isolated central eigenvalues for no-gap and gap cases."
function plot_isolated_spectra(case_nogap, case_gap)
    evals_nogap = sort(real(eigvals(case_nogap.H)))
    evals_gap = sort(real(eigvals(case_gap.H)))

    fig, ax = subplots(1, 1, figsize=(6, 4))
    ax.scatter(ones(length(evals_nogap)), evals_nogap, s=14, label="No gap")
    ax.scatter(fill(2.0, length(evals_gap)), evals_gap, s=14, label="Central gap")
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(["No gap", "Gap"])
    ax.set_ylabel(L"E_n")
    ax.set_title("Isolated central spectrum")
    ax.legend(frameon=false)
    ax.grid(true, axis="y", alpha=0.3)
    tight_layout()
    return fig
end



"Plot only the central-gap isolated spectrum with guide lines at ±Δ/2."
function plot_gap_only_spectrum(case_gap, Δ::Float64)
    evals_gap = sort(real(eigvals(case_gap.H)))
    n = 1:length(evals_gap)

    fig, ax = subplots(1, 1, figsize=(6, 4))
    ax.scatter(n, evals_gap, s=18, color="tab:red", label="Gap case eigenvalues")
    ax.axhline(+Δ/2, color="k", linestyle="--", linewidth=1.0, label=L"+\Delta/2")
    ax.axhline(-Δ/2, color="k", linestyle="--", linewidth=1.0, label=L"-\Delta/2")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel(L"E_n")
    ax.set_title("Central-gap isolated spectrum")
    ax.grid(true, axis="y", alpha=0.3)
    ax.legend(frameon=false)
    tight_layout()
    return fig
end

function plot_currents(case_nogap, case_gap)
    fig, axs = subplots(1, 2, figsize=(11, 4), sharey=true)

    axs[1].plot(case_nogap.obs.t, case_nogap.obs.Iα[1, :], label=L"I_L")
    axs[1].plot(case_nogap.obs.t, case_nogap.obs.Iα[2, :], label=L"I_R")
    axs[1].set_title("No gap")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("Current")
    axs[1].legend(frameon=false)
    axs[1].grid(true, alpha=0.3)

    axs[2].plot(case_gap.obs.t, case_gap.obs.Iα[1, :], label=L"I_L")
    axs[2].plot(case_gap.obs.t, case_gap.obs.Iα[2, :], label=L"I_R")
    axs[2].set_title("Central gap")
    axs[2].set_xlabel("t")
    axs[2].legend(frameon=false)
    axs[2].grid(true, alpha=0.3)

    suptitle("Lead currents")
    tight_layout()
    return fig
end

function plot_site_densities(case_nogap, case_gap)
    Nx = case_nogap.p_model.Nx
    i_left = 1
    i_mid = cld(Nx, 2)
    i_right = Nx

    fig, axs = subplots(1, 2, figsize=(11, 4), sharey=true)
    for (ax, ctitle, data) in zip(axs, ["No gap", "Central gap"], [case_nogap, case_gap])
        ax.plot(data.obs.t, data.obs.n_i[i_left, :], label="site 1")
        ax.plot(data.obs.t, data.obs.n_i[i_mid, :], label="site $(i_mid)")
        ax.plot(data.obs.t, data.obs.n_i[i_right, :], label="site $(i_right)")
        ax.set_title(ctitle)
        ax.set_xlabel("t")
        ax.grid(true, alpha=0.3)
    end
    axs[1].set_ylabel(L"n_i(t)")
    axs[1].legend(frameon=false)
    suptitle("Representative site densities")
    tight_layout()
    return fig
end

function plot_final_density_profile(case_nogap, case_gap)
    x = 1:case_nogap.p_model.Nx
    fig, ax = subplots(1, 1, figsize=(6, 4))
    ax.plot(x, case_nogap.obs.n_i[:, end], "o-", label="No gap")
    ax.plot(x, case_gap.obs.n_i[:, end], "s-", label="Central gap")
    ax.set_xlabel("site index")
    ax.set_ylabel(L"n_i(t_{final})")
    ax.set_title("Final density profile")
    ax.legend(frameon=false)
    ax.grid(true, alpha=0.3)
    tight_layout()
    return fig
end

function main()
    cfg = init_params_blocks_1d_twochannel()

    println("Running Example 4 (1D two-channel benchmark)")
    println("Nx=$(cfg.Nx), Ny=$(cfg.Ny), Nσ=$(cfg.Nσ), N_orb=$(cfg.N_orb)")

    case_nogap = run_case_blocks_1d_twochannel(cfg;
                                               eps_c_center=cfg.eps_c_lead,
                                               eps_v_center=cfg.eps_v_lead,
                                               label="no-gap")

    case_gap = run_case_blocks_1d_twochannel(cfg;
                                             eps_c_center=+cfg.Δ/2,
                                             eps_v_center=-cfg.Δ/2,
                                             label="central-gap")

    println("mean I_L (no-gap)    = $(mean(case_nogap.obs.Iα[1, :]))")
    println("mean I_L (gap)       = $(mean(case_gap.obs.Iα[1, :]))")

    mkpath("examples/data")
    @save "examples/data/example4_1d_twochannel_benchmark.jl2" cfg case_nogap case_gap

    plot_lead_dispersion(cfg)
    plot_bloch_dispersion_comparison(cfg)
    plot_isolated_spectra(case_nogap, case_gap)
    plot_gap_only_spectrum(case_gap, cfg.Δ)
    plot_currents(case_nogap, case_gap)
    plot_site_densities(case_nogap, case_gap)
    plot_final_density_profile(case_nogap, case_gap)

    println("Saved: examples/data/example4_1d_twochannel_benchmark.jl2")
end

main()
