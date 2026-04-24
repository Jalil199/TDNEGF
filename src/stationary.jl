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
### Standard Landauer transmission T(E): Σ evaluated at absolute energy E (Meir-Wingreen).
### H must NOT be shifted. Only the Fermi functions carry the μ_α dependence.
function transmission_std(E::Float64, H::Matrix{ComplexF64};
                          γ::Float64=1.0, γc::Float64=1.0,
                          Ny::Int=2, Nσ::Int=2, η::Float64=1e-8)
    dim = size(H, 1)
    Id  = Matrix{ComplexF64}(I, dim, dim)
    SL  = ΣL_tot(complex(E, η); γ=γ, γc=γc, Nx=div(dim, Ny*Nσ), Ny=Ny, Nσ=Nσ)
    SR  = ΣR_tot(complex(E, η); γ=γ, γc=γc, Nx=div(dim, Ny*Nσ), Ny=Ny, Nσ=Nσ)
    Gr  = inv((E + 1im*η)*Id - H - SL - SR)
    GL  = 1im*(SL - SL')
    GR  = 1im*(SR - SR')
    return real(tr(GL * Gr * GR * Gr'))
end

### Linear-response conductance G(E_F) = (1/2π·δV) ∫ T(E)·[f(E,μL)-f(E,μR)] dE
### Standard Meir-Wingreen: H unshifted, Σ(E) at absolute energy.
function landauer_conductance(E_F::Float64;
                              δV::Float64=0.01, β::Float64=40.0,
                              γ::Float64=1.0, γc::Float64=1.0,
                              Nx::Int=1, Ny::Int=2, Nσ::Int=2, N_orb::Int=1,
                              n_pts::Int=4000, η::Float64=1e-8)
    H  = build_H_ab(; Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, γ=γ, γso=0.0+0.0im)
    μL = E_F + δV/2;  μR = E_F - δV/2
    ε_grid = range(-max(6.0, abs(E_F)+3), max(6.0, abs(E_F)+3); length=n_pts)
    dε = step(ε_grid);  Isum = 0.0
    for ε in ε_grid
        T  = transmission_std(Float64(ε), H; γ=γ, γc=γc, Ny=Ny, Nσ=Nσ, η=η)
        fL = abs(β*(ε-μL)) > 500 ? (ε < μL ? 1.0 : 0.0) : 1.0/(1.0+exp(β*(ε-μL)))
        fR = abs(β*(ε-μR)) > 500 ? (ε < μR ? 1.0 : 0.0) : 1.0/(1.0+exp(β*(ε-μR)))
        Isum += T*(fL - fR)*dε
    end
    return Isum / (2π * δV)
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
               Np::Int, Nx::Int ,Ny::Int , Nσ::Int ,N_orb::Int ; Rtail::Float64=1e12, η = 1e-3)
    
    z_p, R_p = pade_poles(Np)   # z_p = -im*ξ, R_p = -η
    dim = Nx * Ny * Nσ
    one = Matrix{ComplexF64}(I, dim,dim  )
    
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

