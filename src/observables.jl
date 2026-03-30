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
    t::Vector{T}        = Vector{T}(undef, 0)           # (N_tmax,)
    n_i::Matrix{T}      = Matrix{T}(undef, 0, 0)        # (N_sites, N_tmax)
    σx_i::Array{T,3}    = Array{T,3}(undef, 0, 0, 0)    # (N_sites, 3, N_tmax)  (x,y,z)
    ### Equilibrium spin density
    σx_i_eq::Array{T,3}    = Array{T,3}(undef, 0, 0, 0) # (N_sites, 3, N_tmax)  (x,y,z) 
    sx_i::Array{T,3}    = Array{T,3}(undef, 0, 0, 0)    # (N_sites, 3, N_tmax)  (x,y,z)

    Iα::Matrix{T}       = Matrix{T}(undef, 0, 0)        # (N_leads, N_tmax)     lead charge current
    Iαx::Array{T,3}     = Array{T,3}(undef, 0, 0, 0)    # (N_leads, 3, N_tmax)  lead spin current (x,y,z)

    idx::Int            = 0                             # index associates with the iteration (it allow us to
                                                        # run over the time slices of the observables)
end

#### This function Initiallizes the observables 
function ObservablesTDNEGF(p::ModelParamsTDNEGF; N_tmax::Int, N_leads::Int, T::Type{<:AbstractFloat}=Float64)
    ObservablesTDNEGF{T}(;
        t      = Vector{T}(undef, N_tmax),
        n_i    = Matrix{T}(undef, p.N_sites, N_tmax),
        σx_i   = Array{T,3}(undef, p.N_sites, 3, N_tmax),
        σx_i_eq   = Array{T,3}(undef, p.N_sites, 3, N_tmax),
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
### this function calculates local charge density from the global density matrix `ρ_ab`.
### It is backend-independent and accepts both `pointer(...)` and `pointer_blocks(...)` views.


@inline function _accumulate_spin_trace(Bloc, σx, σy, σz, N_loc::Int)
    N_spin = size(σx, 1)
    (N_spin == 2 && size(σx, 2) == 2) || throw(ArgumentError("Pauli matrices must be 2×2 in spin space"))
    N_loc % N_spin == 0 || throw(ArgumentError("N_loc must be divisible by spin dimension"))
    N_orb = N_loc ÷ N_spin

    sx = 0.0; sy = 0.0; sz = 0.0
    @inbounds for orb in 0:(N_orb - 1)
        base = orb * N_spin
        for aσ in 1:N_spin, bσ in 1:N_spin
            a = base + aσ
            b = base + bσ
            ρba = Bloc[b, a]
            sx += real(σx[aσ, bσ] * ρba)
            sy += real(σy[aσ, bσ] * ρba)
            sz += real(σz[aσ, bσ] * ρba)
        end
    end
    return sx, sy, sz
end

@inline function obs_n_i!(dv, p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)
    #### It updates the iterative element associated with temporal index (idx)
    #### the updated index is moved to evolution loop 
    it = obs.idx#obs.idx = it
    ρ = dv.ρ_ab  # both pointer paths expose `ρ_ab` with the same field name
    N_loc = p.N_loc
    @inbounds for i in 1:p.N_sites
        a0 = (i - 1) * N_loc + 1
        b0 = i * N_loc
        ρloc = @view ρ[a0:b0, a0:b0] # view of the local positional subspace of the density matrix
        s = 0.0
        @inbounds for a in 1:N_loc
            s += real(ρloc[a,a]) # the density should be a real number 
        end
        obs.n_i[i, it] = s
    end
    return nothing
end

### Spin density
### This local observable only depends on `ρ_ab`, so it is backend-independent.
@inline function obs_σ_i!(dv, p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)
    it = obs.idx
    ρ  = dv.ρ_ab
    σx, σy, σz = p.σ_x, p.σ_y, p.σ_z
    
    N_loc = p.N_loc
    @inbounds for i in 1:p.N_sites
        a0 = (i - 1) * N_loc + 1
        b0 = i * N_loc
        ρloc = @view ρ[a0:b0, a0:b0]

        sx, sy, sz = _accumulate_spin_trace(ρloc, σx, σy, σz, N_loc)
        obs.σx_i[i,1,it] = sx
        obs.σx_i[i,2,it] = sy
        obs.σx_i[i,3,it] = sz
    end
    return nothing
end

@inline function obs_σ_i_eq!(ρ::Matrix{ComplexF64} , p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)
    ### This function updates the spin density from an arbitraty density matrix 
    it = obs.idx
    #ρ  = dv.ρ_ab
    σx, σy, σz = p.σ_x, p.σ_y, p.σ_z
    N_loc = p.N_loc
    @inbounds for i in 1:p.N_sites
        a0 = (i - 1) * N_loc + 1
        b0 = i * N_loc
        ρloc = @view ρ[a0:b0, a0:b0]

        sx, sy, sz = _accumulate_spin_trace(ρloc, σx, σy, σz, N_loc)
        obs.σx_i_eq[i,1,it] = sx
        obs.σx_i_eq[i,2,it] = sy
        obs.σx_i_eq[i,3,it] = sz
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

"""
    cal_Π_abα(ptr, p_blocks)

Builds per-block Π tensors with the heterogeneous auxiliary layout:
`Π[:,:,α] = (Σ_λ Ψ[:,:,λ]_α) * ξ_α^T`.

The block path reuses preallocated buffers in `ExperimentalBlockRHSParams`
(`Ψ_an` and `Π_abα_obs`) to avoid per-call/per-block allocations.
"""
@inline function cal_Π_abα(ptr::HeterogeneousAuxPointers, p::ExperimentalBlockRHSParams)
    Ns = p.dims_ρ_ab[1]
    Nblocks = length(p.blocks)
    Π_abα = p.Π_abα_obs

    @inbounds for α in 1:Nblocks
        block = p.blocks[α]
        bptr = ptr.blocks[α]
        Ψ_an = p.Ψ_an[α]
        for n in 1:block.Nc, a in 1:Ns
            acc = 0.0 + 0.0im
            @simd for λ in 1:block.N_λ
                acc += bptr.Ψ_anλ[a, n, λ]
            end
            Ψ_an[a, n] = acc
        end
        mul!(@view(Π_abα[:, :, α]), Ψ_an, transpose(block.ξ_an))
    end

    return Π_abα
end

### Lead charge and spin current 
### calculates the lead charge and spin current
@inline function obs_Ixα!(dv::DynamicalVariables,p::ModelParamsTDNEGF, obs::ObservablesTDNEGF)#, site_ranges::Vector{UnitRange{Int64}})
    it = obs.idx
    Π = cal_Π_abα(dv,p)
    #Π  = p.Π_abα #### Note that this is precalculated inside the eom !
    σx, σy, σz = p.σ_x, p.σ_y, p.σ_z
    N_loc = p.N_loc
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
            a0 = (i - 1) * N_loc + 1
            b0 = i * N_loc
            Πloc = @view Πα[a0:b0, a0:b0]
            tx, ty, tz = _accumulate_spin_trace(Πloc, σx, σy, σz, N_loc)
            sx += tx; sy += ty; sz += tz
        end

        obs.Iαx[α,1,it] = 2*sx
        obs.Iαx[α,2,it] = 2*sy
        obs.Iαx[α,3,it] = 2*sz
    end
    return nothing
end

"""
    obs_Ixα!(ptr, p_blocks, obs)

Preferred current observable for the heterogeneous block backend.

`obs.Iα` and `obs.Iαx` are indexed by auxiliary block order
(`p_blocks.blocks`). This may differ from the old rectangular `α` meaning
unless blocks are configured one-per-physical-lead.
"""
@inline function obs_Ixα!(ptr::HeterogeneousAuxPointers, p_blocks::ExperimentalBlockRHSParams, obs::ObservablesTDNEGF)
    p_blocks.obs_N_sites > 0 || throw(ArgumentError("ExperimentalBlockRHSParams is missing observable geometry. Build it with ExperimentalBlockRHSParams(H_ab, blocks, Δ_blocks, p_model)."))
    it = obs.idx
    Π = cal_Π_abα(ptr, p_blocks)
    σx, σy, σz = p_blocks.obs_σ_x, p_blocks.obs_σ_y, p_blocks.obs_σ_z
    Nblocks = length(p_blocks.blocks)

    @inbounds for α in 1:Nblocks
        Πα = @view Π[:, :, α]
        trΠα = 0.0
        for a in 1:size(Πα, 1)
            trΠα += real(Πα[a, a])
        end
        obs.Iα[α, it] = 2 * trΠα

        sx = 0.0; sy = 0.0; sz = 0.0
        for i in 1:p_blocks.obs_N_sites
            r = p_blocks.obs_site_ranges[i]
            Πloc = @view Πα[r, r]
            tx, ty, tz = _accumulate_spin_trace(Πloc, σx, σy, σz, p_blocks.obs_N_loc)
            sx += tx; sy += ty; sz += tz
        end
        obs.Iαx[α, 1, it] = 2 * sx
        obs.Iαx[α, 2, it] = 2 * sy
        obs.Iαx[α, 3, it] = 2 * sz
    end
    return nothing
end

"""
    obs_Ixα!(ptr, p_blocks, p_model, obs)

Compatibility-only overload for block current observables.

Use this only when `p_blocks` was built without observable metadata via
`ExperimentalBlockRHSParams(H_ab, blocks, Δ_blocks)`. The preferred path is
`obs_Ixα!(ptr, p_blocks, obs)` with `p_blocks` constructed as
`ExperimentalBlockRHSParams(H_ab, blocks, Δ_blocks, p_model)`.
"""
@inline function obs_Ixα!(ptr::HeterogeneousAuxPointers, p_blocks::ExperimentalBlockRHSParams, p_model::ModelParamsTDNEGF, obs::ObservablesTDNEGF)
    it = obs.idx
    Π = cal_Π_abα(ptr, p_blocks)
    σx, σy, σz = p_model.σ_x, p_model.σ_y, p_model.σ_z
    N_loc = p_model.N_loc
    Nblocks = length(p_blocks.blocks)

    @inbounds for α in 1:Nblocks
        Πα = @view Π[:, :, α]
        trΠα = 0.0
        for a in 1:size(Πα, 1)
            trΠα += real(Πα[a, a])
        end
        obs.Iα[α, it] = 2 * trΠα

        sx = 0.0; sy = 0.0; sz = 0.0
        for i in 1:p_model.N_sites
            a0 = (i - 1) * N_loc + 1
            r = a0:(a0 + N_loc - 1)
            Πloc = @view Πα[r, r]
            tx, ty, tz = _accumulate_spin_trace(Πloc, σx, σy, σz, N_loc)
            sx += tx; sy += ty; sz += tz
        end
        obs.Iαx[α, 1, it] = 2 * sx
        obs.Iαx[α, 2, it] = 2 * sy
        obs.Iαx[α, 3, it] = 2 * sz
    end
    return nothing
end
