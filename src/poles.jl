# Function to generate pole expansions 
# Pade poles for the fermi function 
function pade_poles(M::Int)
    ### Calculates the residue and poles of the pade decomposition 
    @assert M ≥ 1 "The number of poles should be greater or equal than 1"
    M=2*M
    # Symmetric tridiagonal with off-diagonals b_n = 1 / (2*sqrt(4n^2 - 1))
    b = [1.0/(2*sqrt(4n^2 - 1.0)) for n in 1:M-1]
    T = SymTridiagonal(zeros(M), b)        # main diagonal zeros, off-diagonals b
    F = eigen(T)                           # F.values::Vector, F.vectors::Matrix
    λ = F.values
    V = F.vectors
    # keep positive eigenvalues
    pos = findall(>(0.0), λ)
    ξ = 1 ./λ[pos]                       # poles (imag-axis locations)
    η = norm.(V[1, pos]).^2 ./ (4.0*λ[pos].^2)             # weights = (first component)^2
    return -1im*ξ, -η #### note the -im by convention 
end

### Rebuild fermi function from poles and residues 
ferm_test(x::ComplexF64,η_k::Vector{ComplexF64},
            ξ_k::Vector{ComplexF64},β::Float64) = 0.5- sum(η_k/β.*(1 ./( 1im*ξ_k/β .+ x ) .+  1 ./( -1im*ξ_k/β .+ x )   )) 

function _load_precomputed_complex_table(path::AbstractString)
    data = readdlm(path)
    size(data, 2) == 2 || throw(ArgumentError("expected two-column complex table at $path"))
    return complex.(data[:, 1], data[:, 2])
end

# Tables whose residues are stored already multiplied by 2π (legacy Lorentzian fit).
const _PREFACTORED_TABLES = ("Lorentz",)

function _load_λ1_square_data(N_λ1::Int; name::Union{Nothing,AbstractString} = nothing)
    path = joinpath(dirname(@__DIR__), "data")
    # Backward compatibility: without a name, N_λ1 = 31 means the legacy Lorentzian
    # table and anything else the minimal-pole semicircle tables.
    name === nothing && (name = N_λ1 == 31 ? "Lorentz" : "Semicircle")
    z_path = joinpath(path, "z_$(name)_N$(N_λ1).txt")
    r_path = joinpath(path, "R_$(name)_N$(N_λ1).txt")
    if !(isfile(z_path) && isfile(r_path))
        available = sort(unique([m.captures[1] * " N=" * m.captures[2] for f in readdir(path)
                                 for m in (match(r"^z_(.+)_N(\d+)\.txt$", f),) if m !== nothing]))
        throw(ArgumentError("No pole table for $(name) with N_λ1=$(N_λ1) in $(path). " *
                            "Available: $(join(available, ", ")). " *
                            "Generate one with tools/generate_poles.py (minimal pole method) " *
                            "or tools/fit_lorentzians.py (Lorentzians)."))
    end
    scale = name in _PREFACTORED_TABLES ? 1.0 : 2pi
    return load_poles_files(z_path, r_path; residue_scale = scale)
end

"""
    load_poles_files(z_path, R_path; residue_scale = 2π)

Load spectral poles `zλ1` and residues `Rλ1` from two-column (real, imag) text
files written by `tools/generate_poles.py`. Residues are multiplied by
`residue_scale`; the default 2π converts a fitted spectral function A(ω) into
Γ(ω) = 2π A(ω). Returns `(Rλ1, zλ1)`.
"""
function load_poles_files(z_path::AbstractString, R_path::AbstractString; residue_scale::Real = 2pi)
    zλ1 = _load_precomputed_complex_table(z_path)
    Rλ1 = _load_precomputed_complex_table(R_path) .* residue_scale
    length(zλ1) == length(Rλ1) || throw(ArgumentError("pole and residue tables have different lengths"))
    return Rλ1, zλ1
end

"""
    load_poles_square(N_λ1 = 49, N_λ2 = 20; name = nothing)

Spectral poles of the lead plus `N_λ2` Padé poles of the Fermi function.
Returns `(Rλ, zλ)`.

The spectral table is `data/z_<name>_N<N_λ1>.txt` / `R_<name>_N<N_λ1>.txt`, so
`name` selects the fit:

- `"Semicircle"`: minimal pole method (`tools/generate_poles.py`), e.g. N_λ1 = 49;
- `"Lorentz"`: legacy sum of Lorentzians, N_λ1 = 31;
- `"SemicircleLorentz"` or any other name: tables written by
  `tools/generate_poles.py` or `tools/fit_lorentzians.py`.

Without `name`, N_λ1 = 31 loads `"Lorentz"` and any other N_λ1 loads
`"Semicircle"` (the behavior of earlier versions). Tables outside `data/` are
loaded with [`load_poles_files`](@ref).
"""
function load_poles_square(N_λ1::Int = 49, N_λ2::Int = 20; name::Union{Nothing,AbstractString} = nothing)
    zλ2, Rλ2 = pade_poles(N_λ2)
    Rλ1, zλ1 = _load_λ1_square_data(N_λ1; name)
    return [Rλ1; Rλ2], [zλ1; zλ2]
end
