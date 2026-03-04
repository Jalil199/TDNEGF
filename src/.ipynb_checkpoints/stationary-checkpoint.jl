### This module contains main function to get stationary calcualtions 

function Selfenergy(;E,H0l,H1l,Hlc,eta, epsilon=1e-12)
    "Calculation of the selfenergy. 
    The parameters needed are
    H0L : Supercell
    HLC: Matrix connecting the central system with the leads
    H1L : Hopping between super cells"
    one = Diagonal(ones(size(H0l)[1]))
    A0 = H1l'
    B0 = H1l
    C0 = (E + 1im*eta)*one - H0l
    D0 = (E +1im*eta)*one - H0l
    invD0 = inv(D0)
    while norm(A0) > epsilon
        #println(1)
        C = C0 - A0*invD0*B0
        D = D0 - A0*invD0*B0 - B0*invD0*A0
        A = A0*invD0*A0
        B = B0*invD0*B0
        ###################################
        A0 = A
        B0 = B
        C0 = C
        D0 = D
        invD0 =inv(D0)
    end
    g_L = inv(C0) ### Surface green function of the lead 
    Self = Hlc'*g_L*Hlc
end


### Diagonal component of the self energy for rectangular lattice
function Σn_anal(ϵ::Union{ComplexF64,Float64}; n::Int, γ::Float64=1.0, γc::Float64=1.0, Ny::Int=5)
    ϵ_n = -2 * γ * cos(n * π / (Ny + 1))
    z   = ϵ - ϵ_n
    Δ   = z^2 - 4 * γ^2
    s   = sqrt(complex(Δ))
    if imag(s) < 0 || (isapprox(imag(s), 0.0; atol=1e-14) && real(s) * real(z) < 0)
        s = -s
    end
    Σ = (γc^2 / (2 * γ^2)) * (z - s)
    return Σ
end
### Matrix components of the analitical expression for the self energy     

function Σ_anal(ϵ::Union{ComplexF64,Float64}; γ::Float64 = 1.,γc::Float64= 1.0 ,Ny::Int=2,Nσ::Int = 2,N_orb::Int = 1)
    # We use the unitary transformation to go back to the lattice basis
    U_in(i,n) = sin(n*i*pi/(Ny+1))*sqrt(2/(Ny+1))
    one_σ = Matrix{ComplexF64}(I, Nσ, Nσ)
    
    Σ = zeros(ComplexF64, Ny*Nσ, Ny*Nσ)
    for i in 1:Ny,j in 1:Ny
        i_idx = get_sub(i, Nσ*N_orb)
        j_idx = get_sub(j, Nσ*N_orb)

        for n_s in 1:Ny
             Σ[i_idx,j_idx] .+= U_in(i,n_s)*(Σn_anal(ϵ;n=n_s,γ=γ,γc=γc,Ny=Ny)*one_σ)*U_in(j,n_s)
        end
    end
    return  Σ
end


### Build objects  in the total dimension 

## Calculation of the total self energy 
function Σ_tot(ϵ::Union{Float64,ComplexF64}; γ::Float64=1., γc::Float64=1.0, Nx::Int=1, Ny::Int=2, Nσ::Int=2,N_orb::Int =1)
    ## Total embedding self-energy matrix for left and right leads
    dim = Nx * Ny * Nσ * N_orb
    Σ_t = zeros(ComplexF64, dim, dim)
    # At the moment we compute the self energy only for equivalent rectangular lattices
    Σ_L = Σ_anal(ϵ; γ=γ, γc=γc, Ny=Ny, Nσ=Nσ)
    Σ_R = Σ_anal(ϵ; γ=γ, γc=γc, Ny=Ny, Nσ=Nσ)
    # Left contact: first x-slice
    iL = 1 : Ny * Nσ
    # Right contact: last x-slice
    iR = dim - Ny * Nσ + 1 : dim
    Σ_t[iL, iL] .+= Σ_L
    Σ_t[iR, iR] .+= Σ_R
    return Σ_t
end

function ΣL_tot(ϵ::Union{Float64,ComplexF64}; γ::Float64=1., γc::Float64=1.0, Nx::Int=1, Ny::Int=2, Nσ::Int=2)
    ## Total embedding self-energy matrix for left and right leads
    dim = Nx * Ny * Nσ
    Σ_t = zeros(ComplexF64, dim, dim)
    Σ_L = Σ_anal(complex(ϵ); γ=γ, γc=γc, Ny=Ny, Nσ=Nσ)
    iL = 1 : Ny * Nσ
    Σ_t[iL, iL] .+= Σ_L
    return Σ_t
end
function ΣR_tot(ϵ::Union{Float64,ComplexF64}; γ::Float64=1., γc::Float64=1.0, Nx::Int=1, Ny::Int=2, Nσ::Int=2)
    ## Total embedding self-energy matrix for left and right leads
    dim = Nx * Ny * Nσ
    Σ_t = zeros(ComplexF64, dim, dim)
    Σ_R = Σ_anal(ϵ; γ=γ, γc=γc, Ny=Ny, Nσ=Nσ)
    iR = dim - Ny * Nσ + 1 : dim
    Σ_t[iR, iR] .+= Σ_R
    return Σ_t
end
### Now The retarded component GF should be computed 
function Gr(ϵ::Union{Float64,ComplexF64} ,H::Matrix{ComplexF64}, Σ;η=0.02)
    dims= size(H)[1]
    one = Matrix{ComplexF64}(I, dims,dims  )
    G = inv((ϵ + 1im*η )*one - H - Σ(complex(ϵ)) )
    return G
end
### Now The advanced component GF should be computed 
function Ga(ϵ::Union{Float64,ComplexF64} ,H::Matrix{ComplexF64}, Σ;η=1e-2)
    #η = 1e-2
    dims= size(H)[1]
    one = Matrix{ComplexF64}(I, dims,dims  )
    
    G = inv( ((ϵ + 1im*η )*one - H - Σ(complex(ϵ)) )' )
    return G
end

### Build the density matrix in Eq. for the instantaneus Hamiltonian
function ρ_eq(ϵ_f::Float64, β::Float64, H::Matrix{ComplexF64},
               Np::Int ,Ny::Int , Nσ::Int ,N_orb::Int ; Rtail::Float64=1e12, η = 1e-3)
    
    z_p, R_p = pade_poles(Np)   # z_p = -im*ξ, R_p = -η
    dim = Nx * Ny * Nσ
    
    ### Calculation of the GF based on the instantaneous Hamiltonian of the central system 
    
    ### Note that parameters are for typicall quasi-1D rectangular lattice
    Σ_T(ϵ)  = Σ_tot(ϵ; γ=1.0,γc=1.0 , Nx = Nx, Ny=Ny , Nσ = Nσ )
    ### We build the total Green function 
    Gr(ϵ)   = inv((ϵ + 1im*η )*one - H - Σ_T(complex(ϵ)) )
    Gtmp = Gr(ϵ_f + conj(z_p[1]) / β)
    
    acc = zeros(eltype(Gtmp), size(Gtmp))
    for i in eachindex(z_p)
        pole = ϵ_f + conj(z_p[i]) / β   # = μ + iξ/β
        acc .+= (-R_p[i]) .* Gr(pole)   # = η * G
    end
    X = (2im / β) .* acc
    ρ = (X .- X') ./ (2im)
    moment = 0.5im * Rtail .* Gr(1im * Rtail)
    return ρ + moment
end

