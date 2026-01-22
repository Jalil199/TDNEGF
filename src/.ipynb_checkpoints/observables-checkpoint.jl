### Functions to calculate the observables for the TDNEGF code 

### This function get the sub indexes (a:b) asociated with a particular degree of freedom
@inline function get_sub(idx::Int, sub_dim::Int)
    ### Get the sub index range 
    a = (idx-1)*sub_dim + 1
    b = idx*sub_dim
    return a:b
end

### Structure containig the main observables calculated by TDNEGF code 
Base.@kwdef mutable struct ObservablesTDNEGF{T<:AbstractFloat}
    t::Vector{T}        = Vector{T}(undef, 0)         # (N_tmax,)
    n_i::Matrix{T}      = Matrix{T}(undef, 0, 0)      # (N_sites, N_tmax)
    σx_i::Array{T,3}    = Array{T,3}(undef, 0, 0, 0)  # (N_sites, 3, N_tmax)  (x,y,z)
    sx_i::Array{T,3}    = Array{T,3}(undef, 0, 0, 0)  # (N_sites, 3, N_tmax)  (x,y,z)

    Iα::Matrix{T}       = Matrix{T}(undef, 0, 0)      # (N_leads, N_tmax)     lead charge current
    Iαx::Array{T,3}     = Array{T,3}(undef, 0, 0, 0)  # (N_leads, 3, N_tmax)  lead spin current (x,y,z)

    idx::Int            = 0                           # index associates with the iteration (it allow us to
                                                      # run over the time slices of the observables)
end

#### This function Initiallizes the observables 
function ObservablesTDNEGF(p::ModelParamsTDNEGF; N_tmax::Int, N_leads::Int, T::Type{<:AbstractFloat}=Float64)
    ObservablesTDNEGF{T}(;
        t      = Vector{T}(undef, N_tmax),
        n_i    = Matrix{T}(undef, p.N_sites, N_tmax),
        σx_i   = Array{T,3}(undef, p.N_sites, 3, N_tmax),
        sx_i   = Array{T,3}(undef, p.N_sites, 3, N_tmax),
        Iα     = Matrix{T}(undef, N_leads, N_tmax),
        Iαx    = Array{T,3}(undef, N_leads, 3, N_tmax),
        idx    = 0
    )
end

### Classical spin density
### this function calculates the classical density evaluated at all the classical lattice 
@inline function obs_s_i!(S::Matrix{SVector{3,Float64}}, p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)
     it = obs.idx
     @inbounds for i in 1:length(S) # this corresponds to a single index (i) associates with the 2d indexes (i,j)
        a, b = ij_from_linear(i, p.Ny)   # (i) to (i,j) 
        Si = S[a,b]
        
        obs.sx_i[i,1,it] = Si[1]
        obs.sx_i[i,2,it] = Si[2]
        obs.sx_i[i,3,it] = Si[3]
    end
    return nothing 
end

### Charge density
### this function calculaes the local charge density of the electrons from the density matrix 
### Note that this function returns directly all the local charge densities.
@inline function obs_n_i!(dv::DynamicalVariables, p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)
    #### It updates the iterative element associated with temporal index (idx)
    #### the updated index is moved to evolution loop 
    it = obs.idx#obs.idx = it
    ρ = dv.ρ_ab  # view of the density matrix 
    site_ranges = [get_sub(i, p.N_loc) for i in 1:p.N_sites] #sub indexes associate to the spatial-local degrees of freedom 
    
    @inbounds for i in 1:p.N_sites
        # Note that i 
        r = site_ranges[i]
        ρloc = @view ρ[r, r] # view of the local positional subspace of the density matrix
        s = 0.0
        @inbounds for a in 1:length(r)
            s += real(ρloc[a,a]) # the density should be a real number 
        end
        obs.n_i[i, it] = s
    end
    return nothing
end

### Spin density 
@inline function obs_σ_i!(dv::DynamicalVariables, p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)
    it = obs.idx
    ρ  = dv.ρ_ab
    σx, σy, σz = p.σ_x, p.σ_y, p.σ_z
    
    site_ranges = [get_sub(i, p.N_loc) for i in 1:p.N_sites] 
    
    @inbounds for i in 1:p.N_sites
        r = site_ranges[i]
        ρloc = @view ρ[r, r]

        sx=0.0; sy=0.0; sz=0.0
        @inbounds for a in 1:p.N_loc, b in 1:p.N_loc
            ρba = ρloc[b,a]   # tr(σ ρ) = Σ_ab σ_ab ρ_ba
            sx += real(σx[a,b] * ρba)
            sy += real(σy[a,b] * ρba)
            sz += real(σz[a,b] * ρba)
        end
        obs.σx_i[i,1,it] = sx
        obs.σx_i[i,2,it] = sy
        obs.σx_i[i,3,it] = sz
    end
    return nothing
end

@inline function cal_Π_abα(dv::DynamicalVariables,p::ModelParamsTDNEGF)

    ### Dimensions
    Nα = p.Nα
    Ns = p.Ns
    Nc = p.Nc
    N_λ = p.N_λ
    (;Ψ_anλα) = dv
    ξ_anα     = p.ξ_anα
    
    # Buffers
    Ψ_anα   = p.Ψ_anα         # Ns × Nc × Nα
    Π_ab    = p.Π_ab          # Ns × Ns
    Π_abα     = p.Π_abα
    # 1A) Ψ_anα = Σ_λ Ψ_anλα
    @inbounds for α in 1:Nα
        for n in 1:Nc
            for a in 1:Ns
                Ψ_anα_anα = zero(eltype(Ψ_anλα))
                @simd for λ in 1:N_λ
                    Ψ_anα_anα += Ψ_anλα[a, n, λ, α]
                end
                Ψ_anα[a, n, α] = Ψ_anα_anα
            end
        end
    end
    ###### calculate Π without allocate 
    # 1B) Π_abα(⋅,⋅,α) = Ψ_anα(⋅,⋅,α) * ξ_anα(⋅,⋅,α)^T
    @inbounds for α in 1:p.Nα
        Ψ_anα_α = @view Ψ_anα[:, :, α]     # Ns × Nc
        ξ_anα_α = @view ξ_anα[:, :, α]     # Ns × Nc
        Π_abα_α = @view Π_abα[:, :, α]     # Ns × Ns
        mul!(Π_abα_α, Ψ_anα_α, transpose(ξ_anα_α))
    end

    # 1C) Π_ab = Σ_α Π_abα[:,:,α]
    fill!(Π_ab, zero(eltype(Π_ab)))
    @inbounds for α in 1:Nα
        Π_abα_α = @view Π_abα[:, :, α]
        @inbounds for a in 1:Ns, b in 1:Ns
            Π_ab[a, b] += Π_abα_α[a, b]
        end
    end    
    return Π_abα
end # end cal_Π_abα

### Lead charge and spin current 
### calculates the lead charge and spin current
@inline function obs_Ixα!(dv::DynamicalVariables,p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)#, site_ranges::Vector{UnitRange{Int64}})
    it = obs.idx
    Π = cal_Π_abα(dv,p)
    #Π  = p.Π_abα #### Note that this is precalculated inside the eom !
    σx, σy, σz = p.σ_x, p.σ_y, p.σ_z
    site_ranges = [get_sub(i, p.N_loc) for i in 1:p.N_sites] #sub indexes associate to the spatial-local degrees of freedom 
    @inbounds for α in 1:p.Nα
        Πα = @view Π[:, :, α]
        # ---- carga: 2 Re Tr[Πα] ----
        trΠα = 0.0
        @inbounds for a in 1:p.Ns
            trΠα += real(Πα[a,a])
        end
        obs.Iα[α, it] = 2*trΠα
        # ---- espín: 2 Re Tr[σloc Πα] por bloques ----
        sx = 0.0; sy = 0.0; sz = 0.0
        @inbounds for i in 1:p.N_sites
            r = site_ranges[i]
            Πloc = @view Πα[r, r]
            tx=0.0; ty=0.0; tz=0.0
            @inbounds for a in 1:p.N_loc, b in 1:p.N_loc
                Πba = Πloc[b,a]
                tx += real(σx[a,b] * Πba)
                ty += real(σy[a,b] * Πba)
                tz += real(σz[a,b] * Πba)
            end
            sx += tx; sy += ty; sz += tz
        end

        obs.Iαx[α,1,it] = 2*sx
        obs.Iαx[α,2,it] = 2*sy
        obs.Iαx[α,3,it] = 2*sz
    end
    return nothing
end




