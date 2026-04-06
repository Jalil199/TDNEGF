### test_selfenergy_timedomain.jl
###
### Two checks for the pole representation of the lead self-energy:
###
###  1. Spectral check: G_rec(ω, R, z) reconstructs the exact semicircle Γ(ω)
###  2. Time-domain check: the pole sum Σ^<(t) = Σ_λ ΣL_nλ exp(-i χ_nλ t)
###     matches the direct numerical Fourier transform
###     Σ^<(t) = ∫ dω/(2π) f(ω) Γ(ω) e^{-iωt}
###
###  Both checks are run for N_λ1 = 49 (semicircle) and N_λ1 = 31 (Lorentz).
###  The 1D case (Ny=1) is used so ϵ_n = 0 and the channel index is trivial.

using Test
using LinearAlgebra
using TDNEGF

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

## Exact retarded self-energy of a semi-infinite 1D tight-binding chain
## with hopping γ and contact hopping γc (surface Green's function * γc^2)
function exact_selfenergy(ϵ::Number; γ::Float64=1.0, γc::Float64=1.0)
    Δ = 4γ^2 - ϵ^2
    if real(Δ) > 0
        Σ = ϵ - im * sqrt(complex(Δ))
    else
        sgn = real(ϵ) ≥ 0 ? 1 : -1
        Σ = ϵ - sgn * sqrt(complex(-Δ))
    end
    return Σ * (γc^2 / (2γ^2))
end

## Level-width function: Γ(ω) = -2 Im(Σ^R(ω))
exact_Γ(ω::Float64; γ::Float64=1.0, γc::Float64=1.0) =
    -2 * imag(exact_selfenergy(ω; γ=γ, γc=γc))

## Fermi function
fermi(ω::Float64; β::Float64) = 1.0 / (1.0 + exp(ω * β))

## Reconstructed Γ from poles (symmetric form, real axis)
function Γ_reconstructed(ω::Float64, Rλ1::Vector{ComplexF64}, zλ1::Vector{ComplexF64})
    g = zero(ComplexF64)
    for i in eachindex(zλ1)
        g += Rλ1[i] / (ω - zλ1[i]) + conj(Rλ1[i]) / (ω - conj(zλ1[i]))
    end
    return real(g)   # Γ(ω) is real
end

## Pole-sum reconstruction of Σ^<(t) for channel n
##   By the residue theorem (closing in the lower half-plane for t > 0):
##     ∫dω/(2π) Σ^<(ω) e^{-iωt} = -i * Σ_{Im(χ)<0} ΣL[n,λ] * exp(-i χ[n,λ] * t)
##   For t < 0 closing in upper half-plane gives the hermitian-conjugate relation.
function ΣL_pole_sum(t::Float64, n::Int,
                     ΣL_nλ::Matrix{ComplexF64},
                     χ_nλ::Matrix{ComplexF64})
    if t >= 0
        return -1im * sum(ΣL_nλ[n, λ] * exp(-1im * χ_nλ[n, λ] * t) for λ in axes(χ_nλ, 2))
    else
        return -conj(-1im * sum(ΣL_nλ[n, λ] * exp(-1im * χ_nλ[n, λ] * (-t)) for λ in axes(χ_nλ, 2)))
    end
end

## Direct numerical Fourier transform of Σ^<(ω) = f(ω) Γ(ω)
##   Σ^<(t) = ∫ dω/(2π) f(ω) Γ(ω) e^{-iωt}
## using trapezoidal rule on [-ω_max, ω_max]
function ΣL_direct_ft(t::Float64, n::Int; β::Float64,
                      ω_max::Float64=8.0, dω::Float64=0.005,
                      γ::Float64=1.0, γc::Float64=1.0, ϵ_n::Float64=0.0)
    ωs  = -ω_max:dω:ω_max
    integrand = [fermi(ω - ϵ_n; β=β) * exact_Γ(ω - ϵ_n; γ=γ, γc=γc) * exp(-1im * ω * t)
                 for ω in ωs]
    return (dω / (2π)) * (sum(integrand) - 0.5 * (integrand[1] + integrand[end]))
end

rel_rmse(x, y) = norm(x .- y) / max(norm(x), 1e-14)
max_abs(x, y)  = maximum(abs.(x .- y))

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------

const β_TEST   = 33.0          # inverse temperature (γ units, ≈ 300 K for γ=1 eV)
const Ny_TEST  = 1             # 1D lead: single transverse mode, ϵ_n = 0
const Nσ_TEST  = 2             # spin degeneracy
const N_orb_TEST = 1
const N_λ2_TEST  = 20          # Padé poles for Fermi function

# tolerances
const TOL_SPECTRAL_N49 = 5e-3  # max pointwise error: semicircle N49 (very good fit)
const TOL_SPECTRAL_N31 = 0.12  # max pointwise error: Lorentz N31 (fit has edge errors)
const TOL_TIMEDOMAIN = 5e-2    # rel_rmse of Σ^<(t) over time window

# time grid for time-domain check
const t_TEST = collect(0.0:0.5:20.0)

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

@testset "Self-energy pole representation" begin

    for (label, N_λ1, tol_spec) in [("Semicircle N49", 49, TOL_SPECTRAL_N49),
                                     ("Lorentz N31",    31, TOL_SPECTRAL_N31)]

        @testset "$label" begin

            # --- load poles ---
            Rλ, zλ = load_poles_square(N_λ1, N_λ2_TEST)
            Rλ1 = Rλ[1:N_λ1]
            zλ1 = zλ[1:N_λ1]

            # --- build ΣL and χ for 1D lead (Ny=1) ---
            ΣL_nλ = build_Σᴸ_nλ(Rλ, zλ, Ny_TEST, Nσ_TEST, N_orb_TEST,
                                  N_λ1, N_λ2_TEST; β=β_TEST, γ=1.0)
            χ_nλ  = build_χ_nλ(zλ, Ny_TEST, Nσ_TEST, N_orb_TEST,
                                N_λ1, N_λ2_TEST; β=β_TEST, γ=1.0)

            Nc = Ny_TEST * Nσ_TEST * N_orb_TEST   # = 2

            # ----------------------------------------------------------
            # Check 1: spectral reconstruction of Γ(ω)
            # ----------------------------------------------------------
            @testset "Spectral reconstruction Γ(ω)" begin
                ωs      = collect(-1.8:0.05:1.8)   # inside the band
                Γ_exact = exact_Γ.(ωs)
                Γ_rec   = [Γ_reconstructed(ω, Rλ1, zλ1) for ω in ωs]

                err = max_abs(Γ_exact, Γ_rec)
                @test err < tol_spec
                println("  [$label] Spectral: max_abs(Γ_exact - Γ_rec) = $(round(err, sigdigits=3))")
            end

            # ----------------------------------------------------------
            # Check 2: time-domain Σ^<(t) — pole sum vs direct FT
            # ----------------------------------------------------------
            @testset "Time-domain Σ^<(t) channel 1" begin
                n = 1   # first channel (Ny=1, Nσ=2 → 2 channels, both with ϵ_n=0)

                ΣL_poles  = [ΣL_pole_sum(t, n, ΣL_nλ, χ_nλ) for t in t_TEST]
                ΣL_direct = [ΣL_direct_ft(t, n; β=β_TEST) for t in t_TEST]

                err_rel = rel_rmse(ΣL_poles, ΣL_direct)
                err_max = max_abs(ΣL_poles, ΣL_direct)

                @test err_rel < TOL_TIMEDOMAIN
                println("  [$label] Time-domain ch1: rel_rmse = $(round(err_rel, sigdigits=3)), max_abs = $(round(err_max, sigdigits=3))")
            end

            @testset "Time-domain Σ^<(t) channel 2" begin
                n = 2

                ΣL_poles  = [ΣL_pole_sum(t, n, ΣL_nλ, χ_nλ) for t in t_TEST]
                ΣL_direct = [ΣL_direct_ft(t, n; β=β_TEST) for t in t_TEST]

                err_rel = rel_rmse(ΣL_poles, ΣL_direct)
                err_max = max_abs(ΣL_poles, ΣL_direct)

                @test err_rel < TOL_TIMEDOMAIN
                println("  [$label] Time-domain ch2: rel_rmse = $(round(err_rel, sigdigits=3)), max_abs = $(round(err_max, sigdigits=3))")
            end

        end  # testset label
    end  # for label

end  # testset
