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

function _load_λ1_square_data(N_λ1::Int)
    path = joinpath(dirname(@__DIR__), "data")
    if N_λ1 == 49
        z_path = joinpath(path, "z_Semicircle_N49.txt")
        r_path = joinpath(path, "R_Semicircle_N49.txt")
        zλ1 = _load_precomputed_complex_table(z_path)
        Rλ1 = _load_precomputed_complex_table(r_path) .* (2pi)
        return Rλ1, zλ1
    elseif N_λ1 == 31
        z_path = joinpath(path, "z_Lorentz_N31.txt")
        r_path = joinpath(path, "R_Lorentz_N31.txt")
        zλ1 = _load_precomputed_complex_table(z_path)
        Rλ1 = _load_precomputed_complex_table(r_path)
        return Rλ1, zλ1
    end
    throw(ArgumentError("There are not precalculated quantities for N_λ1=$N_λ1"))
end

function load_poles_square(N_λ1::Int = 49, N_λ2::Int = 20)
    ### Calculates the fermi poles 
    zλ2, Rλ2 = pade_poles(N_λ2)
    Rλ1, zλ1 = _load_λ1_square_data(N_λ1)
    Rλ = [Rλ1;  Rλ2]
    zλ = [zλ1;  zλ2]
    
    return Rλ, zλ

end
