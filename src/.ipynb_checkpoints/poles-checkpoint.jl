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

function load_poles_square(N_λ1::Int = 49, N_λ2::Int = 20)
    @assert N_λ1==49 "There are not precalculated quantities for this number of poles"
    ### Calculates the fermi poles 
    zλ2, Rλ2 = pade_poles(N_λ2)
    ### Get the precalculated poles from MP external package
    path = joinpath(dirname(@__DIR__))
    data_z = readdlm(path*"/data/z_Semicircle_N49.txt")  # matriz N×2
    data_R = readdlm(path*"/data/R_Semicircle_N49.txt")  # matriz N×2
    ### In the future an option to call MP packge should be added
    #------------------------------------------------------------
    
    #------------------------------------------------------------
    zλ1 = complex.(data_z[:, 1], data_z[:, 2])  # Vector{ComplexF64}
    Rλ1 = complex.(data_R[:, 1], data_R[:, 2]) ; # Vector{ComplexF64};
    Rλ1 = Rλ1*2pi ; ### added by convention 
    Rλ = [Rλ1;  Rλ2]
    zλ = [zλ1;  zλ2]
    
    return Rλ, zλ

end