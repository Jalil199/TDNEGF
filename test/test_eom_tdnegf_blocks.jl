using Test
using TDNEGF
using LinearAlgebra
using Random

function build_equivalent_rhs_setup()
    p_rect = ModelParamsTDNEGF(; Nx = 1, Ny = 1, Nσ = 2, N_orb = 1, Nα = 1, N_λ1 = 1, N_λ2 = 1)

    p_rect.H_ab .= ComplexF64[
        0.2 + 0.0im 0.1 - 0.05im
        0.1 + 0.05im -0.1 + 0.0im
    ]

    p_rect.ξ_anα[:, :, 1] .= ComplexF64[
        1.0 + 0.0im 0.0 + 0.0im
        0.0 + 0.0im 1.0 + 0.0im
    ]

    p_rect.Σᴸ_nλα[:, :, 1] .= ComplexF64[
        0.05 + 0.01im 0.03 + 0.00im
        0.04 - 0.02im 0.02 + 0.01im
    ]
    p_rect.Σᴳ_nλα[:, :, 1] .= ComplexF64[
        0.02 + 0.00im 0.01 + 0.00im
        0.03 + 0.00im 0.015 + 0.00im
    ]
    p_rect.χ_nλα[:, :, 1] .= ComplexF64[
        -0.30 + 0.20im -0.10 + 0.10im
        -0.25 + 0.15im -0.12 + 0.08im
    ]
    p_rect.Δ_α[1] = 0.07 + 0.00im

    p_rect.Γ_nλα .= 1im .* (p_rect.Σᴳ_nλα .- p_rect.Σᴸ_nλα)
    p_rect.χ′_nλα .= conj.(p_rect.χ_nλα)
    p_rect.Σᴸ′_nλα .= conj.(p_rect.Σᴸ_nλα)
    p_rect.Γ′_nλα .= conj.(p_rect.Γ_nλα)

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
    p_blocks = ExperimentalBlockRHSParams(copy(p_rect.H_ab), [block])

    layouts, total_aux = build_selfenergy_aux_layout([block])
    @assert length(layouts) == 1

    Random.seed!(17)
    u_rect = randn(ComplexF64, p_rect.size_u)
    u_blocks = randn(ComplexF64, p_rect.size_ρ_ab + total_aux)
    @assert length(u_rect) == length(u_blocks)
    copyto!(u_blocks, u_rect)

    return p_rect, p_blocks, u_rect, u_blocks
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

@testset "Rectangular and block RHS equivalence regression" begin
    p_rect, p_blocks, u_rect, u_blocks = build_equivalent_rhs_setup()

    du_rect = similar(u_rect)
    du_blocks = similar(u_blocks)

    eom_tdnegf!(du_rect, u_rect, p_rect, 0.0)
    eom_tdnegf_blocks!(du_blocks, u_blocks, p_blocks, 0.0)

    @test p_rect.Γ_nλα ≈ 1im .* (p_rect.Σᴳ_nλα .- p_rect.Σᴸ_nλα)
    @test du_rect ≈ du_blocks rtol = 1e-12 atol = 1e-12

    dt = 1e-4
    t = 0.0
    u_rect_t = copy(u_rect)
    u_blocks_t = copy(u_blocks)

    for _ in 1:5
        u_rect_t = rk4_step(eom_tdnegf!, u_rect_t, p_rect, t, dt)
        u_blocks_t = rk4_step(eom_tdnegf_blocks!, u_blocks_t, p_blocks, t, dt)
        t += dt
    end

    @test u_rect_t ≈ u_blocks_t rtol = 1e-11 atol = 1e-11
end
