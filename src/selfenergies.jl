### Static container for one self-energy block used by the current ξ-based auxiliary backend.
### This is not yet a fully general representation of arbitrary self-energies.
Base.@kwdef struct SelfEnergyBlock
    name::Symbol
    Nc::Int
    N_λ1::Int
    N_λ2::Int
    N_λ::Int
    ΣL_nλ::Matrix{ComplexF64}
    ΣG_nλ::Matrix{ComplexF64}
    χ_nλ::Matrix{ComplexF64}
    ξ_an::Matrix{ComplexF64}
    Δ::ComplexF64
end

function SelfEnergyBlock(
    name::Symbol,
    Nc::Int,
    N_λ1::Int,
    N_λ2::Int,
    N_λ::Int,
    ΣL_nλ::Matrix{ComplexF64},
    ΣG_nλ::Matrix{ComplexF64},
    χ_nλ::Matrix{ComplexF64},
    ξ_an::Matrix{ComplexF64},
    Δ::ComplexF64,
)
    size(ΣL_nλ) == (Nc, N_λ) || throw(ArgumentError("size(ΣL_nλ) must be (Nc, N_λ)"))
    size(ΣG_nλ) == (Nc, N_λ) || throw(ArgumentError("size(ΣG_nλ) must be (Nc, N_λ)"))
    size(χ_nλ) == (Nc, N_λ) || throw(ArgumentError("size(χ_nλ) must be (Nc, N_λ)"))
    size(ξ_an, 2) == Nc || throw(ArgumentError("size(ξ_an, 2) must equal Nc"))

    return SelfEnergyBlock(; name, Nc, N_λ1, N_λ2, N_λ, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an, Δ)
end

function SelfEnergyBlock(
    name::Symbol,
    ΣL_nλ::Matrix{ComplexF64},
    ΣG_nλ::Matrix{ComplexF64},
    χ_nλ::Matrix{ComplexF64},
    ξ_an::Matrix{ComplexF64},
    Δ::ComplexF64,
)
    Nc, N_λ = size(ΣL_nλ)
    N_λ1 = N_λ
    N_λ2 = 0
    return SelfEnergyBlock(name, Nc, N_λ1, N_λ2, N_λ, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an, Δ)
end
