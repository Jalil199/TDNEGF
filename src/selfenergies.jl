### Minimal static container for self-energy blocks
struct SelfEnergyBlock
    name::Symbol
    ΣL_nλ::Matrix{ComplexF64}
    ΣG_nλ::Matrix{ComplexF64}
    χ_nλ::Matrix{ComplexF64}
    ξ_an::Matrix{ComplexF64}
    Δ::ComplexF64
end
