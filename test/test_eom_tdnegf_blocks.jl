using Test
using TDNEGF
using LinearAlgebra
using Random

function build_equivalent_rhs_setup_multiblock()
    p_rect = ModelParamsTDNEGF(; Nx = 1, Ny = 1, Nσ = 2, N_orb = 1, Nα = 2, N_λ1 = 1, N_λ2 = 1)

    p_rect.H_ab .= ComplexF64[
        0.2 + 0.0im 0.1 - 0.05im
        0.1 + 0.05im -0.1 + 0.0im
    ]

    p_rect.ξ_anα[:, :, 1] .= ComplexF64[
        1.0 + 0.0im 0.0 + 0.0im
        0.0 + 0.0im 1.0 + 0.0im
    ]
    p_rect.ξ_anα[:, :, 2] .= ComplexF64[
        0.8 + 0.1im 0.1 + 0.0im
        0.0 - 0.1im 0.9 + 0.0im
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

    p_rect.Σᴸ_nλα[:, :, 2] .= ComplexF64[
        0.02 - 0.03im 0.01 + 0.01im
        0.03 + 0.01im 0.015 - 0.01im
    ]
    p_rect.Σᴳ_nλα[:, :, 2] .= ComplexF64[
        0.01 + 0.00im 0.006 + 0.001im
        0.02 - 0.002im 0.010 + 0.000im
    ]
    p_rect.χ_nλα[:, :, 2] .= ComplexF64[
        -0.18 + 0.06im -0.09 + 0.04im
        -0.22 + 0.08im -0.11 + 0.05im
    ]

    p_rect.Δ_α[1] = 0.07 + 0.00im
    p_rect.Δ_α[2] = -0.03 + 0.00im

    p_rect.Γ_nλα .= 1im .* (p_rect.Σᴳ_nλα .- p_rect.Σᴸ_nλα)
    p_rect.χ′_nλα .= conj.(p_rect.χ_nλα)
    p_rect.Σᴸ′_nλα .= conj.(p_rect.Σᴸ_nλα)
    p_rect.Γ′_nλα .= conj.(p_rect.Γ_nλα)

    block1 = SelfEnergyBlock(
        :left,
        p_rect.Nc,
        p_rect.N_λ1,
        p_rect.N_λ2,
        copy(p_rect.Σᴸ_nλα[:, :, 1]),
        copy(p_rect.Σᴳ_nλα[:, :, 1]),
        copy(p_rect.χ_nλα[:, :, 1]),
        copy(p_rect.ξ_anα[:, :, 1]),
        p_rect.Δ_α[1],
    )
    block2 = SelfEnergyBlock(
        :right,
        p_rect.Nc,
        p_rect.N_λ1,
        p_rect.N_λ2,
        copy(p_rect.Σᴸ_nλα[:, :, 2]),
        copy(p_rect.Σᴳ_nλα[:, :, 2]),
        copy(p_rect.χ_nλα[:, :, 2]),
        copy(p_rect.ξ_anα[:, :, 2]),
        p_rect.Δ_α[2],
    )
    p_blocks = ExperimentalBlockRHSParams(copy(p_rect.H_ab), [block1, block2])

    Random.seed!(17)
    u_rect = randn(ComplexF64, p_rect.size_u)
    u_blocks = zeros(ComplexF64, p_rect.size_ρ_ab + p_blocks.aux_layout.total_size)

    copy_rect_to_blocks!(u_blocks, u_rect, p_rect, p_blocks)

    return p_rect, p_blocks, u_rect, u_blocks
end

function copy_rect_to_blocks!(u_blocks::Vector{ComplexF64}, u_rect::Vector{ComplexF64}, p_rect::ModelParamsTDNEGF, p_blocks::ExperimentalBlockRHSParams)
    rect = TDNEGF.pointer(u_rect, p_rect)
    blk = pointer_blocks(u_blocks, p_blocks.dims_ρ_ab, p_blocks.aux_layout)

    blk.ρ_ab .= rect.ρ_ab

    Nα = p_rect.Nα

    for α in 1:Nα
        blk.blocks[α].Ψ_anλ .= rect.Ψ_anλα[:, :, :, α]
    end

    for α in 1:Nα
        for α_p in 1:Nα
            blk.Ω_pairs[α, α_p].Ω11 .= rect.Ω_nλ1α_nλ1α[:, :, α, :, :, α_p]
            blk.Ω_pairs[α, α_p].Ω12 .= rect.Ω_nλ1α_nλ2α[:, :, α, :, :, α_p]
            blk.Ω_pairs[α, α_p].Ω21 .= rect.Ω_nλ2α_nλ1α[:, :, α, :, :, α_p]
        end
    end
    return nothing
end

function copy_blocks_to_rect!(u_rect::Vector{ComplexF64}, u_blocks::Vector{ComplexF64}, p_rect::ModelParamsTDNEGF, p_blocks::ExperimentalBlockRHSParams)
    rect = TDNEGF.pointer(u_rect, p_rect)
    blk = pointer_blocks(u_blocks, p_blocks.dims_ρ_ab, p_blocks.aux_layout)

    rect.ρ_ab .= blk.ρ_ab
    for α in 1:p_rect.Nα
        rect.Ψ_anλα[:, :, :, α] .= blk.blocks[α].Ψ_anλ
    end
    for α in 1:p_rect.Nα
        for α_p in 1:p_rect.Nα
            rect.Ω_nλ1α_nλ1α[:, :, α, :, :, α_p] .= blk.Ω_pairs[α, α_p].Ω11
            rect.Ω_nλ1α_nλ2α[:, :, α, :, :, α_p] .= blk.Ω_pairs[α, α_p].Ω12
            rect.Ω_nλ2α_nλ1α[:, :, α, :, :, α_p] .= blk.Ω_pairs[α, α_p].Ω21
        end
    end
    return nothing
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

@testset "Rectangular and block RHS equivalence regression (multi-block)" begin
    p_rect, p_blocks, u_rect, u_blocks = build_equivalent_rhs_setup_multiblock()

    du_rect = similar(u_rect)
    du_blocks = similar(u_blocks)

    eom_tdnegf!(du_rect, u_rect, p_rect, 0.0)
    eom_tdnegf_blocks!(du_blocks, u_blocks, p_blocks, 0.0)

    @test p_rect.Γ_nλα ≈ 1im .* (p_rect.Σᴳ_nλα .- p_rect.Σᴸ_nλα)
    for α in 1:p_rect.Nα
        @test p_blocks.Γ[α] ≈ 1im .* (p_blocks.blocks[α].ΣG_nλ .- p_blocks.blocks[α].ΣL_nλ)
    end

    du_blocks_as_rect = similar(du_rect)
    copy_blocks_to_rect!(du_blocks_as_rect, du_blocks, p_rect, p_blocks)
    @test du_rect ≈ du_blocks_as_rect rtol = 1e-12 atol = 1e-12

    dt = 1e-4
    t = 0.0
    u_rect_t = copy(u_rect)
    u_blocks_t = copy(u_blocks)

    for _ in 1:5
        u_rect_t = rk4_step(eom_tdnegf!, u_rect_t, p_rect, t, dt)
        u_blocks_t = rk4_step(eom_tdnegf_blocks!, u_blocks_t, p_blocks, t, dt)
        t += dt
    end

    u_blocks_t_as_rect = similar(u_rect_t)
    copy_blocks_to_rect!(u_blocks_t_as_rect, u_blocks_t, p_rect, p_blocks)

    @test u_rect_t ≈ u_blocks_t_as_rect rtol = 1e-11 atol = 1e-11
end
