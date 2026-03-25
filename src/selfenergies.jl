"""
    SelfEnergyBlock

Static metadata + coefficient tensors for one block in the experimental
block-based auxiliary path.

Shape conventions (all per block):
- `ΣL_nλ`, `ΣG_nλ`, `χ_nλ`: `(Nc, N_λ)`
- `ξ_an`: `(Ns, Nc)` where `Ns` is the system Hilbert-space size
- `N_λ = N_λ1 + N_λ2` splits poles exactly as in the legacy solver's
  `(λ1, λ2)` sectors (`Ω11`, `Ω12`, `Ω21`).

`SelfEnergyBlock` intentionally stores only static structural data.
Dynamic energy shifts (Δ) live in solver-level parameters (`Δ_blocks`) so
bias scans/time-dependent workflows can update Δ without rebuilding blocks.
"""
struct SelfEnergyBlock
    name::Symbol
    Nc::Int
    N_λ1::Int
    N_λ2::Int
    N_λ::Int
    ΣL_nλ::Matrix{ComplexF64}
    ΣG_nλ::Matrix{ComplexF64}
    χ_nλ::Matrix{ComplexF64}
    ξ_an::Matrix{ComplexF64}

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
    )
        Nc > 0 || throw(ArgumentError("Nc must be positive"))
        N_λ1 ≥ 0 || throw(ArgumentError("N_λ1 must be non-negative"))
        N_λ2 ≥ 0 || throw(ArgumentError("N_λ2 must be non-negative"))
        N_λ == N_λ1 + N_λ2 || throw(ArgumentError("N_λ must equal N_λ1 + N_λ2"))
        size(ΣL_nλ) == (Nc, N_λ) || throw(ArgumentError("size(ΣL_nλ) must be (Nc, N_λ)"))
        size(ΣG_nλ) == (Nc, N_λ) || throw(ArgumentError("size(ΣG_nλ) must be (Nc, N_λ)"))
        size(χ_nλ) == (Nc, N_λ) || throw(ArgumentError("size(χ_nλ) must be (Nc, N_λ)"))
        size(ξ_an, 2) == Nc || throw(ArgumentError("size(ξ_an, 2) must equal Nc"))

        return new(name, Nc, N_λ1, N_λ2, N_λ, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an)
    end
end

"""
    SelfEnergyBlock(name, Nc, N_λ1, N_λ2, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an)

Primary constructor for auxiliary blocks.
`N_λ1` and `N_λ2` must be provided explicitly to avoid ambiguous or inconsistent
metadata in the auxiliary backend.
"""
function SelfEnergyBlock(
    name::Symbol,
    Nc::Int,
    N_λ1::Int,
    N_λ2::Int,
    ΣL_nλ::Matrix{ComplexF64},
    ΣG_nλ::Matrix{ComplexF64},
    χ_nλ::Matrix{ComplexF64},
    ξ_an::Matrix{ComplexF64},
)
    _, N_λ = size(ΣL_nλ)
    return SelfEnergyBlock(name, Nc, N_λ1, N_λ2, N_λ, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an)
end

function SelfEnergyBlock(
    name::Symbol,
    N_λ1::Int,
    N_λ2::Int,
    ΣL_nλ::Matrix{ComplexF64},
    ΣG_nλ::Matrix{ComplexF64},
    χ_nλ::Matrix{ComplexF64},
    ξ_an::Matrix{ComplexF64},
)
    Nc, N_λ = size(ΣL_nλ)
    return SelfEnergyBlock(name, Nc, N_λ1, N_λ2, N_λ, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an)
end
