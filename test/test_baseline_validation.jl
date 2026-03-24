using Test
using TDNEGF
using LinearAlgebra
using Random

const BASELINE_SHA = "66fd00e6997a062edeacff9cea95f6da71c2cd6b"

function build_homogeneous_setup_rect()
    p = ModelParamsTDNEGF(; Nx = 1, Ny = 1, Nσ = 2, N_orb = 1, Nα = 1, N_λ1 = 1, N_λ2 = 1)

    p.H_ab .= ComplexF64[
        0.2 + 0.0im 0.1 - 0.05im
        0.1 + 0.05im -0.1 + 0.0im
    ]

    p.ξ_anα[:, :, 1] .= ComplexF64[
        1.0 + 0.0im 0.0 + 0.0im
        0.0 + 0.0im 1.0 + 0.0im
    ]

    p.Σᴸ_nλα[:, :, 1] .= ComplexF64[
        0.05 + 0.01im 0.03 + 0.00im
        0.04 - 0.02im 0.02 + 0.01im
    ]
    p.Σᴳ_nλα[:, :, 1] .= ComplexF64[
        0.02 + 0.00im 0.01 + 0.00im
        0.03 + 0.00im 0.015 + 0.00im
    ]
    p.χ_nλα[:, :, 1] .= ComplexF64[
        -0.30 + 0.20im -0.10 + 0.10im
        -0.25 + 0.15im -0.12 + 0.08im
    ]
    p.Δ_α[1] = 0.07 + 0.00im

    p.Γ_nλα .= 1im .* (p.Σᴳ_nλα .- p.Σᴸ_nλα)
    p.χ′_nλα .= conj.(p.χ_nλα)
    p.Σᴸ′_nλα .= conj.(p.Σᴸ_nλα)
    p.Γ′_nλα .= conj.(p.Γ_nλα)

    return p
end

function build_homogeneous_setup_blocks(p_rect::ModelParamsTDNEGF)
    block = SelfEnergyBlock(
        :homogeneous,
        p_rect.Nc,
        p_rect.N_λ1,
        p_rect.N_λ2,
        p_rect.N_λ,
        copy(p_rect.Σᴸ_nλα[:, :, 1]),
        copy(p_rect.Σᴳ_nλα[:, :, 1]),
        copy(p_rect.χ_nλα[:, :, 1]),
        copy(p_rect.ξ_anα[:, :, 1]),
        p_rect.Δ_α[1],
    )

    return ExperimentalBlockRHSParams(copy(p_rect.H_ab), [block])
end

function load_baseline_module(baseline_sha::AbstractString)
    mod = Module(:BaselineTDNEGF)
    Core.eval(mod, :(using LinearAlgebra))
    Core.eval(mod, :(using StaticArrays))

    types_src = read(`git show $(baseline_sha):src/types.jl`, String)
    eom_src = read(`git show $(baseline_sha):src/eom_tdnegf.jl`, String)

    Base.include_string(mod, types_src, "baseline_types.jl")
    Base.include_string(mod, eom_src, "baseline_eom_tdnegf.jl")
    return mod
end

function rk4_step(rhs!, u::Vector{ComplexF64}, p, t::Float64, dt::Float64)
    k1 = similar(u)
    k2 = similar(u)
    k3 = similar(u)
    k4 = similar(u)
    utmp = similar(u)

    rhs!(k1, u, p, t)
    @. utmp = u + 0.5 * dt * k1
    rhs!(k2, utmp, p, t + 0.5 * dt)

    @. utmp = u + 0.5 * dt * k2
    rhs!(k3, utmp, p, t + 0.5 * dt)

    @. utmp = u + dt * k3
    rhs!(k4, utmp, p, t + dt)

    unext = similar(u)
    @. unext = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return unext
end

@testset "Baseline validation against pre-refactor commit" begin
    baseline = load_baseline_module(BASELINE_SHA)

    p_current = build_homogeneous_setup_rect()
    p_baseline = baseline.ModelParamsTDNEGF(; Nx = 1, Ny = 1, Nσ = 2, N_orb = 1, Nα = 1, N_λ1 = 1, N_λ2 = 1)

    p_baseline.H_ab .= p_current.H_ab
    p_baseline.ξ_anα .= p_current.ξ_anα
    p_baseline.Σᴸ_nλα .= p_current.Σᴸ_nλα
    p_baseline.Σᴳ_nλα .= p_current.Σᴳ_nλα
    p_baseline.χ_nλα .= p_current.χ_nλα
    p_baseline.Δ_α .= p_current.Δ_α
    p_baseline.Γ_nλα .= 1im .* (p_baseline.Σᴳ_nλα .- p_baseline.Σᴸ_nλα)
    p_baseline.χ′_nλα .= conj.(p_baseline.χ_nλα)
    p_baseline.Σᴸ′_nλα .= conj.(p_baseline.Σᴸ_nλα)
    p_baseline.Γ′_nλα .= conj.(p_baseline.Γ_nλα)

    @test p_current.Γ_nλα == p_baseline.Γ_nλα
    @test p_current.χ′_nλα == p_baseline.χ′_nλα
    @test p_current.Σᴸ′_nλα == p_baseline.Σᴸ′_nλα
    @test p_current.Γ′_nλα == p_baseline.Γ′_nλα

    Random.seed!(17)
    u0 = randn(ComplexF64, p_current.size_u)

    du_current = similar(u0)
    du_baseline = similar(u0)

    eom_tdnegf!(du_current, u0, p_current, 0.0)
    baseline.eom_tdnegf!(du_baseline, u0, p_baseline, 0.0)

    @test du_current ≈ du_baseline rtol = 1e-12 atol = 1e-12

    dt = 1e-4
    t = 0.0
    u_current = copy(u0)
    u_baseline = copy(u0)

    for _ in 1:5
        u_current = rk4_step(eom_tdnegf!, u_current, p_current, t, dt)
        u_baseline = rk4_step(baseline.eom_tdnegf!, u_baseline, p_baseline, t, dt)
        t += dt
    end

    @test u_current ≈ u_baseline rtol = 1e-11 atol = 1e-11

    p_blocks = build_homogeneous_setup_blocks(p_current)
    du_blocks = similar(u0)
    eom_tdnegf_blocks!(du_blocks, u0, p_blocks, 0.0)

    @test du_blocks ≈ du_baseline rtol = 1e-12 atol = 1e-12

    u_blocks = copy(u0)
    t = 0.0
    for _ in 1:5
        u_blocks = rk4_step(eom_tdnegf_blocks!, u_blocks, p_blocks, t, dt)
        t += dt
    end
    @test u_blocks ≈ u_baseline rtol = 1e-11 atol = 1e-11
end

@testset "Invariant checks for homogeneous block path" begin
    p_rect = build_homogeneous_setup_rect()
    p_blocks = build_homogeneous_setup_blocks(p_rect)

    Random.seed!(123)
    u = randn(ComplexF64, p_rect.size_u)

    ρ = reshape(view(u, p_rect.idx_ρ_ab), p_rect.dims_ρ_ab)
    ρ .= 0.5 .* (ρ .+ adjoint(ρ))

    du_rect = similar(u)
    du_blocks = similar(u)
    eom_tdnegf!(du_rect, u, p_rect, 0.0)
    eom_tdnegf_blocks!(du_blocks, u, p_blocks, 0.0)

    dρ_rect = reshape(view(du_rect, p_rect.idx_ρ_ab), p_rect.dims_ρ_ab)
    dρ_blocks = reshape(view(du_blocks, p_rect.idx_ρ_ab), p_rect.dims_ρ_ab)

    @test dρ_rect ≈ adjoint(dρ_rect) rtol = 1e-12 atol = 1e-12
    @test dρ_blocks ≈ adjoint(dρ_blocks) rtol = 1e-12 atol = 1e-12

    layouts, total_aux = build_selfenergy_aux_layout(p_blocks.blocks)
    @test total_aux == p_rect.size_u - p_rect.size_ρ_ab

    ptr = pointer_blocks(copy(u), p_blocks.dims_ρ_ab, layouts)
    ptr_rect = TDNEGF.pointer(copy(u), p_rect)

    @test ptr.ρ_ab == ptr_rect.ρ_ab
    @test ptr.blocks[1].Ψ_anλ == ptr_rect.Ψ_anλα[:, :, :, 1]
    @test ptr.blocks[1].Ω11 == ptr_rect.Ω_nλ1α_nλ1α[:, :, 1, :, :, 1]
    @test ptr.blocks[1].Ω12 == ptr_rect.Ω_nλ1α_nλ2α[:, :, 1, :, :, 1]
    @test ptr.blocks[1].Ω21 == ptr_rect.Ω_nλ2α_nλ1α[:, :, 1, :, :, 1]
end
