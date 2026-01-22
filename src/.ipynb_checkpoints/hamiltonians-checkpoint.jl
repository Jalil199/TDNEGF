### Functions to generate hamiltonians for the active region in an square lattice 
###################module HamiltonianSquare
σ_0 = @SMatrix [1.0 0.0; 0.0 1.0]
σ_x = @SMatrix [0.0 1.0; 1.0 0.0]
σ_y = @SMatrix [0.0 -1im; 1im 0.0]
σ_z = @SMatrix [1.0 0.0; 0.0 -1.0]

#### Builds the super cell Hamiltonian 
function build_blocks(;Ny::Int=1,Nσ::Int=2,N_orb::Int=1,γ::Float64=1,γso::ComplexF64=0.0 + 1im*0.0)
    "Creates the building blocks for a general nx x ny square lattice "
    N_b::Int = Ny*Nσ*N_orb # We include the spin degree of freedom 
    ######
    H0  = zeros(ComplexF64,N_b,N_b)
    T   = zeros(ComplexF64,N_b,N_b)
    I_y = Matrix{ComplexF64}(I,Ny,Ny)
    ######
    Ty  = diagm(-1 =>  ones(Ny-1) )
    T0  = Ty⊗(-γ*σ_0 - 1im*γso*σ_x)
    H0 .= T0 + T0' #-Bz*kron(One_y, σ_z)
    ######
    T .= (I_y)⊗(-γ*σ_0 + γso*1im*σ_y)
    return H0, T
end
#### Builds the Hamiltonian 
function build_H_ab(;Nx::Int=1,Ny::Int=1,Nσ::Int=2,N_orb::Int=1,γ::Float64=1.0,γso::ComplexF64=0.0 + 1im*0.0)
    "This function builds the central hamiltonian for an square lattice"
    Ns::Int  = Nx*Ny*Nσ*N_orb
    HC  = zeros(ComplexF64,Ns,Ns)
    I_x =  Matrix{ComplexF64}(I,Nx,Nx)
    #One_x = Diagonal(ones(nx))
    H0,T = build_blocks(;Ny, Nσ, N_orb, γ, γso)
    Tx   = diagm( -1 =>  ones(Nx-1))⊗T 
    HC   = I_x⊗H0 + Tx + Tx'
    return HC
end

# returns a single index from a tuple, assuming that N_orb =1
@inline linear_index(i::Int, j::Int, Ny::Int) = (i - 1) * Ny + j

# returns a tuple (i,j) from a single index l, assuming that N_orb =1
@inline function ij_from_linear(l::Int, Ny::Int)
    i = (l - 1) ÷ Ny + 1
    j = (l - 1) % Ny + 1
    return i, j
end

### This function updates the Hamiltonian (t -> t+delta t) assuming we include a time dependent  
### sd coupling, e.g. local magnetic moments.
@inline function update_H_e!(p::ModelParamsTDNEGF, site_ranges::Vector{UnitRange{Int64}},
                            S::Matrix{SVector{3,Float64}},
                            j_sd::Float64)
    H0 = p.H0_ab
    H  = p.H_ab
    @assert p.N_loc == 2 "Orbital degrees of freedom are not taken into acount"
    σx, σy, σz = p.σ_x, p.σ_y, p.σ_z
    #@assert p.N_loc == 2
    #i = 1  #####
    @inbounds for l in 1:length(site_ranges)#p.N_sites
        a, b = ij_from_linear(l, p.Ny)   # (x,y) to (i,j) ######## Aqui esta el problema?
        #i += 1 #######
        r = site_ranges[l]               # longitud 2
        Sx = S[a,b][1];  Sy = S[a,b][2];  Sz = S[a,b][3]
        # Block 2×2: H[r,r] = H0[r,r] - j_sd*(S·σ)
        # Note that this version is larger but doesnt introduce allocations
        H[r[1], r[1]] = H0[r[1], r[1]] - j_sd*(Sx*σx[1,1] + Sy*σy[1,1] + Sz*σz[1,1])
        H[r[1], r[2]] = H0[r[1], r[2]] - j_sd*(Sx*σx[1,2] + Sy*σy[1,2] + Sz*σz[1,2])
        H[r[2], r[1]] = H0[r[2], r[1]] - j_sd*(Sx*σx[2,1] + Sy*σy[2,1] + Sz*σz[2,1])
        H[r[2], r[2]] = H0[r[2], r[2]] - j_sd*(Sx*σx[2,2] + Sy*σy[2,2] + Sz*σz[2,2])
    end
    return nothing
end

######################end # module HamiltonianSquare