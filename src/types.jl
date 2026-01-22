# Here is defined most of the structures with the parameters and dynamical variables 

#### Structure with the parameters of the system 
Base.@kwdef struct ModelParamsTDNEGF
    ### Structure storing all the parameters of the system 
    # Sizes of the dimension
    Nx::Int = 2
    Ny::Int = 2
    N_sites::Int = Nx*Ny
    Nσ::Int = 2
    N_orb::Int = 1
    Nα::Int = 2
    Ns::Int = Nx*Ny*Nσ*N_orb
    N_loc::Int = Nσ*N_orb ### Local degrees of freedom 
    # Number of channels 
    Nc::Int = Ny*Nσ*N_orb       
    # Number of poles in MPM decomposition 
    N_λ1::Int
    # Number of poles in pade for fermi function 
    N_λ2::Int
    # Total number of poles 
    N_λ::Int = N_λ1 + N_λ2
    ### 
    dims_Ω_nλ1α_nλ1α::NTuple{6, Int64} = (Nc, N_λ1, Nα, Nc, N_λ1, Nα)
    dims_Ω_nλ1α_nλ2α::NTuple{6, Int64} = (Nc, N_λ1, Nα, Nc, N_λ2, Nα)
    dims_Ω_nλ2α_nλ1α::NTuple{6, Int64} = (Nc, N_λ2, Nα, Nc, N_λ1, Nα)
    dims_Ψ_anλα::NTuple{4, Int64}      = (Ns, Nc, N_λ,  Nα )
    dims_ρ_ab::NTuple{2, Int64}        = (Ns, Ns)
    ###
    size_Ω_nλ1α_nλ1α::Int64 = prod(dims_Ω_nλ1α_nλ1α)
    size_Ω_nλ1α_nλ2α::Int64 = prod(dims_Ω_nλ1α_nλ2α)
    size_Ω_nλ2α_nλ1α::Int64 = prod(dims_Ω_nλ2α_nλ1α)
    size_Ψ_anλα::Int64      = prod(dims_Ψ_anλα)
    size_ρ_ab::Int64        = prod(dims_ρ_ab)
    size_u::Int64           = size_ρ_ab + size_Ψ_anλα + size_Ω_nλ1α_nλ1α + size_Ω_nλ1α_nλ2α + size_Ω_nλ2α_nλ1α
    ### 
    idx_ρ_ab::UnitRange{Int64}         = 1:size_ρ_ab
    idx_Ψ_anλα::UnitRange{Int64}       = size_ρ_ab + 1 : size_Ψ_anλα + size_ρ_ab 
    idx_Ω_nλ1α_nλ1α::UnitRange{Int64}  = size_ρ_ab + size_Ψ_anλα + 1 :  size_ρ_ab + size_Ψ_anλα + size_Ω_nλ1α_nλ1α
    idx_Ω_nλ1α_nλ2α::UnitRange{Int64}  = size_ρ_ab + size_Ψ_anλα + size_Ω_nλ1α_nλ1α + 1 : size_ρ_ab + size_Ψ_anλα + size_Ω_nλ1α_nλ1α +         size_Ω_nλ1α_nλ2α
    idx_Ω_nλ2α_nλ1α::UnitRange{Int64}  = size_ρ_ab + size_Ψ_anλα + size_Ω_nλ1α_nλ1α + size_Ω_nλ1α_nλ2α + 1 : size_ρ_ab + size_Ψ_anλα + size_Ω_nλ1α_nλ1α + size_Ω_nλ1α_nλ2α + size_Ω_nλ2α_nλ1α
    ### Static precalculated objects
    σ_x = @SMatrix [0.0 1.0; 1.0 0.0]
    σ_y = @SMatrix [0.0 -1im; 1im 0.0]
    σ_z = @SMatrix [1.0 0.0; 0.0 -1.0]
    ξ_anα::Array{ComplexF64,3} = zeros(ComplexF64, Ns, Nc, Nα)
    ###
    H0_ab::Array{ComplexF64,2}   = zeros(ComplexF64, Ns, Ns)
    H_ab::Array{ComplexF64,2}    = zeros(ComplexF64, Ns, Ns)
    # Efective residues and poles 
    χ_nλα::Array{ComplexF64,3}   =  zeros(ComplexF64, Nc, N_λ, Nα)
    Γ_nλα::Array{ComplexF64,3}   =  zeros(ComplexF64, Nc, N_λ, Nα)
    Σᴸ_nλα::Array{ComplexF64,3}  =  zeros(ComplexF64, Nc, N_λ, Nα)
    Σᴳ_nλα::Array{ComplexF64,3}  =  zeros(ComplexF64, Nc, N_λ, Nα)
    Δ_α::Array{ComplexF64,1}          =  zeros(ComplexF64, Nα )
    ### Allocation for dynamical variables as tensors
    ρ_ab::Array{ComplexF64,2}        = zeros(ComplexF64, dims_ρ_ab)
    Ψ_anλα::Array{ComplexF64,4}      = zeros(ComplexF64, dims_Ψ_anλα)
    Ω_nλ1α_nλ1α::Array{ComplexF64,6} = zeros(ComplexF64, dims_Ω_nλ1α_nλ1α)
    Ω_nλ1α_nλ2α::Array{ComplexF64,6} = zeros(ComplexF64, dims_Ω_nλ1α_nλ2α)
    Ω_nλ2α_nλ1α::Array{ComplexF64,6} = zeros(ComplexF64, dims_Ω_nλ2α_nλ1α)
    u::Array{ComplexF64}             = zeros(ComplexF64, size_u)
    ### Another instances of dynamical variables 
    Π_abα::Array{ComplexF64,3}       = zeros(ComplexF64, Ns, Ns, Nα)
    ###----------------- WorkSpace---------------------- 
    Hρ::Array{ComplexF64,2}          = zeros(ComplexF64, Ns, Ns)
    Ψ_anα::Array{ComplexF64,3}       = zeros(ComplexF64, Ns, Nc, Nα)
    K_anα::Array{ComplexF64,3}       = zeros(ComplexF64, Ns, Nc, Nα)
    Π_ab::Array{ComplexF64,2}        = zeros(ComplexF64, Ns, Ns)
    tmp_λ1::Vector{ComplexF64}       = zeros(ComplexF64, N_λ1)  # dot1_λ1
    tmp_λ1p::Vector{ComplexF64}      = zeros(ComplexF64, N_λ1)  # dot2_λ1p
    tmp_λ2::Vector{ComplexF64}       = zeros(ComplexF64, N_λ2)  # dot3_λ2
    tmp_λ2p::Vector{ComplexF64}      = zeros(ComplexF64, N_λ2)  # dot4_λ2p
    tmp_Ψ_vec::Vector{ComplexF64}    = zeros(ComplexF64, Ns)
    #### New temp prealocations for Ψ calculation
    HΨ_anλα::Array{ComplexF64,4}   = zeros(ComplexF64, dims_Ψ_anλα)# mismo tamaño que Ψ
    ρξ_anα::Array{ComplexF64,3}    = zeros(ComplexF64, Ns,Nc,Nα )# Ns × Nc × Nα
    χ′_nλα::Array{ComplexF64,3}    =  zeros(ComplexF64, Nc, N_λ, Nα)# conj(χ_nλα), precalculado
    Σᴸ′_nλα ::Array{ComplexF64,3}  =  zeros(ComplexF64, Nc, N_λ, Nα) # conj(Σᴸ_nλα), precalculado
    Γ′_nλα::Array{ComplexF64,3}    =  zeros(ComplexF64, Nc, N_λ, Nα)# conj(Γ_nλα), precalculado
end

#### Structure with allocated memory to dynamical Variables
Base.@kwdef struct DynamicalVariables{
    Tρ<:AbstractArray{ComplexF64,2},
    TΨ<:AbstractArray{ComplexF64,4},
    TΩ<:AbstractArray{ComplexF64,6}
}
    ### This function should be checked in order to specify more clearly the type of Array (to optimize te functions)
    ρ_ab::Tρ
    Ψ_anλα ::TΨ
    ## Omega for a general initial condition
    #Ω_nλα_nλα::TΩ
    ## Omega for contact initial condition 
    Ω_nλ1α_nλ1α::TΩ
    Ω_nλ1α_nλ2α::TΩ
    Ω_nλ2α_nλ1α::TΩ
end

#### Generates a view object of the Dynamical variables 
@inline function pointer(vec::Vector{ComplexF64}, p::ModelParamsTDNEGF)
    ### Recturns the allocated memory from the original vectorized object
    ### with the dyncamil variables 
    ρ_ab = reshape(view(vec,p.idx_ρ_ab ), p.dims_ρ_ab)
    Ψ_anλα = reshape(view(vec,p.idx_Ψ_anλα ),p.dims_Ψ_anλα )
    ### partition of Ω (note that we separete it into 3 components)
    Ω_nλ1α_nλ1α = reshape(view(vec,p.idx_Ω_nλ1α_nλ1α), p.dims_Ω_nλ1α_nλ1α )
    Ω_nλ1α_nλ2α = reshape(view(vec,p.idx_Ω_nλ1α_nλ2α), p.dims_Ω_nλ1α_nλ2α )
    Ω_nλ2α_nλ1α = reshape(view(vec,p.idx_Ω_nλ2α_nλ1α ), p.dims_Ω_nλ2α_nλ1α )
    
    return DynamicalVariables(ρ_ab, Ψ_anλα, Ω_nλ1α_nλ1α, Ω_nλ1α_nλ2α, Ω_nλ2α_nλ1α)
end

