using Test
using TDNEGF
using LinearAlgebra

# Standard Landauer conductance for a clean Ny=2, Nσ=2 quantum wire.
# Σ^r(E) evaluated at absolute energy E (Meir-Wingreen convention).
# Quantization steps: T=4 for |E_F|<1, T=2 for 1<|E_F|<3, T=0 for |E_F|>3.

const NX_LAN = 1
const NY_LAN = 2
const Nσ_LAN = 2
const N_ORB_LAN = 1
const γ_LAN  = 1.0
const β_LAN  = 40.0
const η_LAN  = 1e-9

function _T_std(E::Float64)
    H = TDNEGF.build_H_ab(; Nx=NX_LAN, Ny=NY_LAN, Nσ=Nσ_LAN, N_orb=N_ORB_LAN,
                            γ=γ_LAN, γso=0.0+0.0im)
    dim = size(H, 1)
    Id  = Matrix{ComplexF64}(I, dim, dim)
    SL  = TDNEGF.ΣL_tot(complex(E, η_LAN); γ=γ_LAN, γc=γ_LAN, Nx=NX_LAN, Ny=NY_LAN, Nσ=Nσ_LAN)
    SR  = TDNEGF.ΣR_tot(complex(E, η_LAN); γ=γ_LAN, γc=γ_LAN, Nx=NX_LAN, Ny=NY_LAN, Nσ=Nσ_LAN)
    Gr  = inv((E + 1im*η_LAN)*Id - H - SL - SR)
    GL  = 1im*(SL - SL')
    GR  = 1im*(SR - SR')
    return real(tr(GL * Gr * GR * Gr'))
end

@testset "Standard Landauer transmission quantization (Ny=2, Nσ=2)" begin
    # Band center: all 4 channels open → T=4
    @test _T_std(0.0)  ≈ 4.0  atol=1e-4
    # Between first and second step: 2 channels open → T=2
    @test _T_std(2.0)  ≈ 2.0  atol=1e-4
    @test _T_std(-2.0) ≈ 2.0  atol=1e-4
    # Outside bandwidth: T≈0
    @test _T_std(4.0)  ≈ 0.0  atol=1e-4
    @test _T_std(-4.0) ≈ 0.0  atol=1e-4
end

@testset "Standard Landauer conductance G = T/(2π) at step centers" begin
    # G at E_F=0 should be 4/(2π) ≈ 0.6366
    G0 = TDNEGF.landauer_conductance(0.0; β=β_LAN, Nx=NX_LAN, Ny=NY_LAN, Nσ=Nσ_LAN)
    @test G0 ≈ 4.0/(2π)  atol=1e-4

    # G at E_F=2.0 should be 2/(2π) ≈ 0.3183 (inside band, 1 channel)
    G2 = TDNEGF.landauer_conductance(2.0; β=β_LAN, Nx=NX_LAN, Ny=NY_LAN, Nσ=Nσ_LAN)
    @test G2 ≈ 2.0/(2π)  atol=5e-4

    # G at E_F=-2.0 symmetric
    Gm2 = TDNEGF.landauer_conductance(-2.0; β=β_LAN, Nx=NX_LAN, Ny=NY_LAN, Nσ=Nσ_LAN)
    @test Gm2 ≈ G2  atol=1e-6
end
