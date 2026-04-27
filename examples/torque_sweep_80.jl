#!/usr/bin/env julia

# Small frozen-spin torque sweep.
#
# Conventions follow examples/conductance_verification.jl:
# - H is not globally shifted.
# - Per-lead χ, Σᴸ, Σᴳ are rebuilt with μL/μR.
# - Δ_blocks carries only the bias drop [+V/2, -V/2].
#
# The classical spin is treated as one macrospin coupled uniformly to all
# central sites of the validated Nx=1, Ny=2 wire. The electronic spin density
# is summed over those central sites before forming T = j_sd S × δσ.

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__)))

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Statistics
using Printf
using DelimitedFiles

const Nx = 1
const Ny = 2
const Nσ = 2
const N_orb = 1
const N_λ1 = 49
const N_λ2 = 30
const β = 40.0
const γ = 1.0
const γso = 0.0 + 0.0im
const E_F = 0.0
const j_sd = 0.05
const t_max = 600.0
const n_save = 20

const V_list = [0.0, 0.05]
const u_grid = collect(range(-1.0, 1.0; length=5))   # u = cos(θ)
const φ_grid = collect(range(0.0, 2π; length=9))[1:end-1]

const OUT_DIR = joinpath(dirname(@__DIR__), "examples", "data")
mkpath(OUT_DIR)
const OUT_CSV = joinpath(OUT_DIR, "torque_sweep_80.csv")

@inline function spin_from_uφ(u::Float64, φ::Float64)
    s⊥ = sqrt(max(0.0, 1.0 - u^2))
    return SVector{3,Float64}(s⊥ * cos(φ), s⊥ * sin(φ), u)
end

@inline function cross3(a::SVector{3,Float64}, b::SVector{3,Float64})
    return SVector{3,Float64}(
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
end

function make_macrospin_matrix(S::SVector{3,Float64})
    Smat = Matrix{SVector{3,Float64}}(undef, Nx, Ny)
    fill!(Smat, S)
    return Smat
end

function build_problem(V::Float64, S::SVector{3,Float64}, shared)
    (; Rλ, zλ, ξ_anL, ξ_anR, H0, site_ranges) = shared
    μL = E_F + V / 2
    μR = E_F - V / 2

    χ_L = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μL)
    χ_R = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μR)
    ΣL_L = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μL)
    ΣG_L = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μL)
    ΣL_R = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μR)
    ΣG_R = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μR)

    p_model = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
                                Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
    p_model.H0_ab .= H0
    p_model.H_ab .= H0

    blocks = [
        SelfEnergyBlock(:left, p_model.Nc, N_λ1, N_λ2, ΣL_L, ΣG_L, χ_L, ξ_anL),
        SelfEnergyBlock(:right, p_model.Nc, N_λ1, N_λ2, ΣL_R, ΣG_R, χ_R, ξ_anR),
    ]
    Δ_blocks = ComplexF64[+V / 2, -V / 2]
    p_blocks = ExperimentalBlockRHSParams(copy(H0), blocks, Δ_blocks, p_model)

    update_H_e!(p_blocks, H0, site_ranges, make_macrospin_matrix(S), j_sd; Ny=Ny)
    p_model.H_ab .= p_blocks.H_ab

    ρ0 = ρ_eq(E_F, β, p_blocks.H_ab, N_λ2, Nx, Ny, Nσ, N_orb)
    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)
    u0[1:p_blocks.size_ρ_ab] .= vec(ρ0)

    return p_model, p_blocks, u0
end

function run_point(idx::Int, V::Float64, u::Float64, φ::Float64, shared)
    θ = acos(clamp(u, -1.0, 1.0))
    S = spin_from_uφ(u, φ)
    p_model, p_blocks, u0 = build_problem(V, S, shared)

    t_save = collect(range(0.8 * t_max, t_max; length=n_save))
    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_max), p_blocks)
    sol = solve(prob, Vern7();
                reltol=1e-8, abstol=1e-10,
                dense=false, save_everystep=false,
                saveat=t_save)

    obs = ObservablesTDNEGF(p_model; N_tmax=length(sol.t), N_leads=2)
    obs.t .= sol.t
    for (it, ut) in enumerate(sol.u)
        obs.idx = it
        ptr = pointer_blocks(ut, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        obs_σ_i!(ptr, p_model, obs)
        obs_Ixα!(ptr, p_blocks, obs)
    end

    σ_sum_t = dropdims(sum(obs.σx_i; dims=1); dims=1) # 3 × Nt
    σ_mean = SVector{3,Float64}(
        mean(view(σ_sum_t, 1, :)),
        mean(view(σ_sum_t, 2, :)),
        mean(view(σ_sum_t, 3, :)),
    )
    T_total = j_sd * cross3(S, σ_mean)
    I_L = mean(obs.Iα[1, :])
    I_R = mean(obs.Iα[2, :])
    I_balance = abs(I_L + I_R)
    I_relstd = std(obs.Iα[1, :]) / (abs(I_L) + 1e-14)

    return (
        idx=idx, V=V, u=u, θ=θ, φ=φ,
        Sx=S[1], Sy=S[2], Sz=S[3],
        σx=σ_mean[1], σy=σ_mean[2], σz=σ_mean[3],
        Tx=T_total[1], Ty=T_total[2], Tz=T_total[3],
        IL=I_L, IR=I_R, Ibalance=I_balance, Irelstd=I_relstd,
    )
end

function main()
    println("Threads disponibles: ", Threads.nthreads())
    println("Total corridas: ", length(V_list) * length(u_grid) * length(φ_grid))
    println("Salida: ", OUT_CSV)

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    H0 = build_H_ab(; Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, γ=γ, γso=γso)
    ξ_anL = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=1, y_coup=1:Ny)
    ξ_anR = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=Nx, y_coup=1:Ny)
    p_ref = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
                              Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
    site_ranges = [get_sub(i, p_ref.N_loc) for i in 1:p_ref.N_sites]
    shared = (; Rλ, zλ, ξ_anL, ξ_anR, H0, site_ranges)

    jobs = Tuple{Int,Float64,Float64,Float64}[]
    idx = 0
    for V in V_list, u in u_grid, φ in φ_grid
        idx += 1
        push!(jobs, (idx, V, u, φ))
    end

    results = Vector{Any}(undef, length(jobs))
    progress = Threads.Atomic{Int}(0)
    started = time()

    Threads.@threads for j in eachindex(jobs)
        idx, V, u, φ = jobs[j]
        result = run_point(idx, V, u, φ, shared)
        results[j] = result
        done = Threads.atomic_add!(progress, 1) + 1
        @printf("[%3d/%3d] V=%6.3f u=%+.3f φ=%6.3f  IL=% .6e  relstd=%.3e  elapsed=%.1fs\n",
                done, length(jobs), V, u, φ, result.IL, result.Irelstd, time() - started)
        flush(stdout)
    end

    sort!(results; by=r -> r.idx)
    header = ["idx" "V" "u" "theta" "phi" "Sx" "Sy" "Sz" "sigma_x" "sigma_y" "sigma_z" "T_x" "T_y" "T_z" "I_L" "I_R" "I_balance" "I_relstd"]
    data = Matrix{Any}(undef, length(results), length(header))
    for (i, r) in enumerate(results)
        data[i, :] .= (r.idx, r.V, r.u, r.θ, r.φ, r.Sx, r.Sy, r.Sz,
                       r.σx, r.σy, r.σz, r.Tx, r.Ty, r.Tz,
                       r.IL, r.IR, r.Ibalance, r.Irelstd)
    end
    writedlm(OUT_CSV, vcat(header, data), ",")
    println("Barrido terminado. CSV: ", OUT_CSV)
end

main()
