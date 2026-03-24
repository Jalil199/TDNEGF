using Test
using TDNEGF

function build_common_inputs()
    N_λ1 = 49
    N_λ2 = 2
    p = ModelParamsTDNEGF(; Nx = 2, Ny = 1, Nσ = 2, N_orb = 1, Nα = 2, N_λ1 = N_λ1, N_λ2 = N_λ2)

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    H_ab = build_H_ab(; Nx = p.Nx, Ny = p.Ny, Nσ = p.Nσ, N_orb = p.N_orb, γ = 1.0, γso = 0.0 + 0.0im)

    Σᴸ_nλ = build_Σᴸ_nλ(Rλ, zλ, p.Ny, p.Nσ, p.N_orb, p.N_λ1, p.N_λ2; β = 5.0, γ = 1.0)
    Σᴳ_nλ = build_Σᴳ_nλ(Rλ, zλ, p.Ny, p.Nσ, p.N_orb, p.N_λ1, p.N_λ2; β = 5.0, γ = 1.0)
    χ_nλ = build_χ_nλ(zλ, p.Ny, p.Nσ, p.N_orb, p.N_λ1, p.N_λ2; β = 5.0, γ = 1.0)

    ξ_anL = build_ξ_an(p.Nx, p.Ny, p.Nσ, p.N_orb; xcol = 1, y_coup = 1:p.Ny)
    ξ_anR = build_ξ_an(p.Nx, p.Ny, p.Nσ, p.N_orb; xcol = p.Nx, y_coup = 1:p.Ny)

    return (; H_ab, Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anL, ξ_anR)
end

function init_manual_path(common)
    p = ModelParamsTDNEGF(; Nx = 2, Ny = 1, Nσ = 2, N_orb = 1, Nα = 2, N_λ1 = 49, N_λ2 = 2)

    p.H_ab .= common.H_ab
    p.H0_ab .= common.H_ab

    p.Σᴸ_nλα[:, :, 1] .= common.Σᴸ_nλ
    p.Σᴸ_nλα[:, :, 2] .= common.Σᴸ_nλ
    p.Σᴳ_nλα[:, :, 1] .= common.Σᴳ_nλ
    p.Σᴳ_nλα[:, :, 2] .= common.Σᴳ_nλ
    p.χ_nλα[:, :, 1] .= common.χ_nλ
    p.χ_nλα[:, :, 2] .= common.χ_nλ
    p.ξ_anα[:, :, 1] .= common.ξ_anL
    p.ξ_anα[:, :, 2] .= common.ξ_anR
    p.Δ_α[1] = 0.5 + 0.0im
    p.Δ_α[2] = -0.5 + 0.0im

    p.Γ_nλα .= 1im .* (p.Σᴳ_nλα .- p.Σᴸ_nλα)
    p.χ′_nλα .= conj.(p.χ_nλα)
    p.Σᴸ′_nλα .= conj.(p.Σᴸ_nλα)
    p.Γ′_nλα .= conj.(p.Γ_nλα)

    return p
end

function init_block_path(common)
    p = ModelParamsTDNEGF(; Nx = 2, Ny = 1, Nσ = 2, N_orb = 1, Nα = 2, N_λ1 = 49, N_λ2 = 2)

    left_block = SelfEnergyBlock(:left, p.N_λ1, p.N_λ2, common.Σᴸ_nλ, common.Σᴳ_nλ, common.χ_nλ, common.ξ_anL, 0.5 + 0.0im)
    right_block = SelfEnergyBlock(:right, p.N_λ1, p.N_λ2, common.Σᴸ_nλ, common.Σᴳ_nλ, common.χ_nλ, common.ξ_anR, -0.5 + 0.0im)

    p.H_ab .= common.H_ab
    p.H0_ab .= common.H_ab

    for (α, block) in enumerate((left_block, right_block))
        p.Σᴸ_nλα[:, :, α] .= block.ΣL_nλ
        p.Σᴳ_nλα[:, :, α] .= block.ΣG_nλ
        p.χ_nλα[:, :, α] .= block.χ_nλ
        p.ξ_anα[:, :, α] .= block.ξ_an
        p.Δ_α[α] = block.Δ
    end

    p.Γ_nλα .= 1im .* (p.Σᴳ_nλα .- p.Σᴸ_nλα)
    p.χ′_nλα .= conj.(p.χ_nλα)
    p.Σᴸ′_nλα .= conj.(p.Σᴸ_nλα)
    p.Γ′_nλα .= conj.(p.Γ_nλα)

    return p
end

@testset "SelfEnergyBlock enforces λ-split consistency" begin
    Nc = 2
    Ns = 4
    ΣL_nλ = zeros(ComplexF64, Nc, 3)
    ΣG_nλ = zeros(ComplexF64, Nc, 3)
    χ_nλ = zeros(ComplexF64, Nc, 3)
    ξ_an = zeros(ComplexF64, Ns, Nc)

    @test_throws ArgumentError SelfEnergyBlock(:bad, Nc, 2, 0, 3, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an, 0.0 + 0.0im)
    @test_throws ArgumentError SelfEnergyBlock(:bad, 2, 2, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an, 0.0 + 0.0im)
end

@testset "SelfEnergyBlock initialization matches manual path" begin
    common = build_common_inputs()
    manual = init_manual_path(common)
    block = init_block_path(common)

    @test block.H_ab == manual.H_ab
    @test block.H0_ab == manual.H0_ab
    @test block.Σᴸ_nλα == manual.Σᴸ_nλα
    @test block.Σᴳ_nλα == manual.Σᴳ_nλα
    @test block.χ_nλα == manual.χ_nλα
    @test block.ξ_anα == manual.ξ_anα
    @test block.Δ_α == manual.Δ_α
    @test block.Γ_nλα == manual.Γ_nλα
    @test block.χ′_nλα == manual.χ′_nλα
    @test block.Σᴸ′_nλα == manual.Σᴸ′_nλα
    @test block.Γ′_nλα == manual.Γ′_nλα
end
