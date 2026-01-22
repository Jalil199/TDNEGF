#### Functions to generate the selfenergy asociated objects
module SelfEnergySquare
    using LinearAlgebra
    
    @inline function get_sub(idx::Int, sub_dim::Int)
        ### Get the sub index range given the internal degrees of 
        ## such indexes 
        a = (idx-1)*sub_dim + 1
        b = idx*sub_dim
        return a:b
    end
    
    @inline function fermi(ϵ::ComplexF64;β::Float64)
        ### Fermi function 
       fermi = 1. / (1. + exp(ϵ*β))
       #ϵ> 36 ? 0.0 : (ϵ < -36 ? 1.0 : 1.0/(1.0 + exp(ϵ)))
    end
    
    ### Eigen energies of an semi-infinite square lattice tape 
    @inline ϵ_n(n::Int,Ny::Int ;γ::Float64 = 1.0) = -2*γ*cos(n*pi/(Ny+1)) 
    
    ### Recontructed analitical continuation from poles and residues using Gull's method 
    function G_rec(z::ComplexF64, Al::Vector{ComplexF64}, xl::Vector{ComplexF64})
        G_z = 0.0 + 0.0im
        for i in 1:length(xl)
            G_z += Al[i] / (z - xl[i]) + conj(Al[i]) / (z - conj(xl[i]) )
        end
        return G_z
    end
    
    ### This function creates an instance of the effective residues 
    ### of the lesser component Σ in the diagonal basis
    function build_Σᴸ_nλ(Rλ::Vector{ComplexF64}, zλ::Vector{ComplexF64}, Ny::Int, Nσ::Int, N_orb::Int, Nλ1::Int, Nλ2::Int ; β::Float64,γ::Float64 = 1.0)
        ## It need two external functions which are: fermi and Γ_r
        # Note that for square lattice selfenergy 
        # all the elements can be builded from the semicircle leveldiwth function  and
        # for both spin up and spin down the system should be repeated for nonmagnetic leads
        Nλ::Int    = Nλ1 + Nλ2
        Nc::Int    = Ny*Nσ*N_orb
        dims_Σᴸ_nλ = (Nc, Nλ)
        Σᴸ_nλ      = zeros(ComplexF64, dims_Σᴸ_nλ)
        ### This function calculates the effetive level-diwth function from the MPM method.
        Γ_r(w) =   G_rec(w, Rλ[1:Nλ1], zλ[1:Nλ1])
        ### 
        for i in 1:Ny
            n_idx = get_sub(i, Nσ*N_orb)
            ### Run over the poles of the fermi function 
            for λ1 in 1:Nλ1
                Σᴸ_nλ[n_idx,λ1] .= Rλ[λ1]*fermi( zλ[λ1]+ϵ_n(i,Ny,γ=γ) ,β=β )
            end
            ### Run over the poles of the fermi function 
            for λ2 in Nλ1+1:Nλ
                Σᴸ_nλ[n_idx,λ2] .= (Rλ[λ2]/β)*Γ_r(zλ[λ2]/β-ϵ_n(i,Ny,γ=γ)) #*4pi#Γ(zλ[λ2],i,Ny)
            end
        end
        return Σᴸ_nλ
    end
    
    ### This function creates an instance of the effective residues 
    ### of the greater component Σ in the diagonal basis
    function build_Σᴳ_nλ(Rλ::Vector{ComplexF64}, zλ::Vector{ComplexF64}, Ny::Int, Nσ::Int, N_orb::Int, Nλ1::Int, Nλ2::Int ;β::Float64,γ::Float64 = 1.0)
        Nλ::Int    = Nλ1 + Nλ2
        Nc::Int    = Ny*Nσ*N_orb
        dims_Σᴳ_nλ = (Nc, Nλ)
        Σᴳ_nλ      = zeros(ComplexF64, dims_Σᴳ_nλ)
        ### This function calculates the effetive level-diwth function from the MPM method.
        Γ_r(w) =   G_rec(w, Rλ[1:Nλ1], zλ[1:Nλ1])
        ### 
        for i in 1:Ny
            n_idx = get_sub(i, Nσ*N_orb)
            ### Run over the poles of the fermi function 
            for λ1 in 1:Nλ1
                Σᴳ_nλ[n_idx,λ1] .= -Rλ[λ1]*(1-fermi( zλ[λ1]+ϵ_n(i,Ny,γ=γ), β=β )) ### Note the beta function 
            end
            ### Run over the poles of the fermi function 
            for λ2 in Nλ1+1:Nλ
                Σᴳ_nλ[n_idx,λ2] .= (Rλ[λ2]/β )*Γ_r(zλ[λ2]/β-ϵ_n(i,Ny,γ=γ))
            end
        end
        return Σᴳ_nλ
    end
    
    ### This functiones generates the effective poles to reconstruct the analytical continuation of a 
    ### generic function 
    function build_χ_nλ(zλ::Vector{ComplexF64}, Ny::Int, Nσ::Int, N_orb::Int, Nλ1::Int, Nλ2::Int; β::Float64,γ::Float64)
        Nλ::Int   = Nλ1 + Nλ2
        Nc::Int   = Ny*Nσ*N_orb
        dims_χ_nλ = (Nc, Nλ)
        χ_nλ      = zeros(ComplexF64, dims_χ_nλ)
    
        for i in 1:Ny
            n_idx = get_sub(i, Nσ*N_orb)   # range of local indexes 
            for n in n_idx
                χ_nλ[n, 1:Nλ1]      .= zλ[1:Nλ1] .+ ϵ_n(i,Ny,γ=γ)
                χ_nλ[n, Nλ1+1:Nλ]   .= zλ[Nλ1+1:Nλ] / β  
            end
        end
        return χ_nλ
    end
    
    ### This function builds the eigen states of the lead in the 
    ### position basis 
    function build_ξ_an(Nx::Int, Ny::Int, Nσ::Int, N_orb::Int;
                               xcol::Int, y_coup::UnitRange{Int64} = 1:Ny)
    
        Nloc::Int = Nσ * N_orb           # DOFs por sitio
        Ns::Int   = Nx * Ny * Nloc
        Nc::Int   = Ny * Nloc            # Ny modos × (spin×orb)
        ξ_an      = zeros(ComplexF64, Ns, Nc)
    
        for y in y_coup
            # global index (xcol, y)
            i = (xcol - 1) * Ny + y
            # range of indexes of site (xcol,y)
            i_idx = get_sub(i, Nloc)  
            for ny_mode in 1:Ny
                amp = sqrt(2/(Ny+1)) * sin(ny_mode * y * pi/(Ny+1))
                for (α, row) in enumerate(i_idx)
                    n = (ny_mode - 1)*Nloc + α   # index of the global channel 
                    ξ_an[row, n] = amp
                end
            end
        end
        return ξ_an
    end
    
    
    export build_Σᴸ_nλ,
           build_Σᴳ_nλ,
           build_χ_nλ,
           build_ξ_an
           
end #module SelEnergySquare