### code for the differential equations using MPM and pade decomposition

@inline function _accumulate_Ψ_anα!(Ψ_anα, Ψ_anλα, Ns, Nc, N_λ, Nα)
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
    return nothing
end

@inline function _build_Π_abα!(Π_abα, Ψ_anα, ξ_anα, Nα)
    @inbounds for α in 1:Nα
        Ψ_anα_α = @view Ψ_anα[:, :, α]
        ξ_anα_α = @view ξ_anα[:, :, α]
        Π_abα_α = @view Π_abα[:, :, α]
        mul!(Π_abα_α, Ψ_anα_α, transpose(ξ_anα_α))
    end
    return nothing
end

@inline function _sum_Π_ab!(Π_ab, Π_abα, Ns, Nα)
    fill!(Π_ab, zero(eltype(Π_ab)))
    @inbounds for α in 1:Nα
        Π_abα_α = @view Π_abα[:, :, α]
        @inbounds for a in 1:Ns, b in 1:Ns
            Π_ab[a, b] += Π_abα_α[a, b]
        end
    end
    return nothing
end

@inline function _rhs_ρ!(dρ_ab, ρ_ab, H_ab, Hρ, Π_ab, Ns)
    mul!(Hρ, H_ab, ρ_ab)
    @inbounds for a in 1:Ns
        @simd for b in 1:Ns
            comm = Hρ[a, b] - conj(Hρ[b, a])
            source = Π_ab[a, b] + conj(Π_ab[b, a])
            dρ_ab[a, b] = -1im * comm + source
        end
    end
    return nothing
end

@inline function _prepare_Ψ_products!(HΨ_anλα, ρξ_anα, H_ab, Ψ_anλα, ρ_ab, ξ_anα, Nc, N_λ, Nα, Ns)
    Ncols_Ψ = Nc * N_λ * Nα
    Ncols_ξ = Nc * Nα

    Ψ_mat = reshape(Ψ_anλα, Ns, Ncols_Ψ)
    HΨ_mat = reshape(HΨ_anλα, Ns, Ncols_Ψ)
    ξ_mat = reshape(ξ_anα, Ns, Ncols_ξ)
    ρξ_mat = reshape(ρξ_anα, Ns, Ncols_ξ)

    mul!(HΨ_mat, H_ab, Ψ_mat)
    mul!(ρξ_mat, ρ_ab, ξ_mat)
    return nothing
end

@inline function _rhs_Ψ_local!(
    dΨ_anλα,
    Ψ_anλα,
    HΨ_anλα,
    ξ_anα,
    ρξ_anα,
    χ′_nλα,
    Σᴸ′_nλα,
    Γ′_nλα,
    Δ_α,
    Ns,
    Nc,
    N_λ,
    Nα,
)
    @inbounds for α in 1:Nα
        Δα = Δ_α[α]
        for n in 1:Nc
            ξ_anα_nα = @view ξ_anα[:, n, α]
            ρξ_anα_nα = @view ρξ_anα[:, n, α]

            for λ in 1:N_λ
                χ′ = χ′_nλα[n, λ, α]
                Σᴸ′ = Σᴸ′_nλα[n, λ, α]
                Γ′ = Γ′_nλα[n, λ, α]

                coef_χΨ = 1im * (χ′ + Δα)
                coef_Σξ = 1im * Σᴸ′
                coef_Γρξ = -Γ′

                Ψ_anλα_nλα = @view Ψ_anλα[:, n, λ, α]
                HΨ_anλα_nλα = @view HΨ_anλα[:, n, λ, α]
                dΨ_anλα_nλα = @view dΨ_anλα[:, n, λ, α]

                @simd for a in 1:Ns
                    dΨ_anλα_nλα[a] =
                        -1im * HΨ_anλα_nλα[a] +
                        coef_χΨ * Ψ_anλα_nλα[a] +
                        coef_Σξ * ξ_anα_nα[a] +
                        coef_Γρξ * ρξ_anα_nα[a]
                end
            end
        end
    end
    return nothing
end

@inline function _rhs_Ψ_Ω!(
    dΨ_anλα,
    Ω_nλ1α_nλ1α,
    Ω_nλ1α_nλ2α,
    Ω_nλ2α_nλ1α,
    ξ_anα,
    tmp_Ψ_a,
    Ns,
    Nc,
    N_λ1,
    N_λ2,
    Nα,
)
    @inbounds for α in 1:Nα
        for n in 1:Nc
            for λ1 in 1:N_λ1
                fill!(tmp_Ψ_a, 0.0 + 0.0im)

                for α_p in 1:Nα
                    for n_p in 1:Nc
                        coeff = 0.0 + 0.0im

                        @simd for λ1_p in 1:N_λ1
                            coeff += Ω_nλ1α_nλ1α[n, λ1, α, n_p, λ1_p, α_p]
                        end
                        @simd for λ2_p in 1:N_λ2
                            coeff += Ω_nλ1α_nλ2α[n, λ1, α, n_p, λ2_p, α_p]
                        end

                        coeff *= -1im
                        ξ_anα_npαp = @view ξ_anα[:, n_p, α_p]
                        @simd for a in 1:Ns
                            tmp_Ψ_a[a] += coeff * ξ_anα_npαp[a]
                        end
                    end
                end

                dΨ_anλα_nλ1α = @view dΨ_anλα[:, n, λ1, α]
                @simd for a in 1:Ns
                    dΨ_anλα_nλ1α[a] += tmp_Ψ_a[a]
                end
            end
        end
    end

    @inbounds for α in 1:Nα
        for n in 1:Nc
            for λ2 in 1:N_λ2
                λ = N_λ1 + λ2
                fill!(tmp_Ψ_a, 0.0 + 0.0im)

                for α_p in 1:Nα
                    for n_p in 1:Nc
                        coeff = 0.0 + 0.0im
                        @simd for λ1_p in 1:N_λ1
                            coeff += Ω_nλ2α_nλ1α[n, λ2, α, n_p, λ1_p, α_p]
                        end
                        coeff *= -1im

                        ξ_anα_npαp = @view ξ_anα[:, n_p, α_p]
                        @simd for a in 1:Ns
                            tmp_Ψ_a[a] += coeff * ξ_anα_npαp[a]
                        end
                    end
                end

                dΨ_anλα_nλα = @view dΨ_anλα[:, n, λ, α]
                @simd for a in 1:Ns
                    dΨ_anλα_nλα[a] += tmp_Ψ_a[a]
                end
            end
        end
    end
    return nothing
end

@inline function _rhs_Ω!(
    dΩ_nλ1α_nλ1α,
    dΩ_nλ1α_nλ2α,
    dΩ_nλ2α_nλ1α,
    Ω_nλ1α_nλ1α,
    Ω_nλ1α_nλ2α,
    Ω_nλ2α_nλ1α,
    Ψ_anλα,
    ξ_anα,
    χ_nλα,
    Γ_nλα,
    χ′_nλα,
    Γ′_nλα,
    Δ_α,
    dot1_λ1,
    dot2_λ1p,
    dot3_λ2,
    dot4_λ2p,
    Nc,
    N_λ1,
    N_λ2,
    Nα,
)
    @inbounds for α in 1:Nα
        Δα = Δ_α[α]
        for α_p in 1:Nα
            Δα_p = Δ_α[α_p]

            for n in 1:Nc
                ξ_anα_nα = @view ξ_anα[:, n, α]
                for n_p in 1:Nc
                    ξ_anα_npαp = @view ξ_anα[:, n_p, α_p]

                    @inbounds for λ1 in 1:N_λ1
                        Ψ_anλα_nλ1α = @view Ψ_anλα[:, n, λ1, α]
                        dot1_λ1[λ1] = dot(ξ_anα_npαp, Ψ_anλα_nλ1α)
                    end

                    @inbounds for λ1_p in 1:N_λ1
                        Ψ_anλα_npλ1pαp = @view Ψ_anλα[:, n_p, λ1_p, α_p]
                        dot_tmp = dot(ξ_anα_nα, Ψ_anλα_npλ1pαp)
                        dot2_λ1p[λ1_p] = conj(dot_tmp)
                    end

                    @inbounds for λ2 in 1:N_λ2
                        Ψ_anλα_nλ2α = @view Ψ_anλα[:, n, N_λ1 + λ2, α]
                        dot3_λ2[λ2] = dot(ξ_anα_npαp, Ψ_anλα_nλ2α)
                    end

                    @inbounds for λ2_p in 1:N_λ2
                        Ψ_anλα_npλ2pαp = @view Ψ_anλα[:, n_p, N_λ1 + λ2_p, α_p]
                        dot_tmp = dot(ξ_anα_nα, Ψ_anλα_npλ2pαp)
                        dot4_λ2p[λ2_p] = conj(dot_tmp)
                    end

                    @inbounds for λ1 in 1:N_λ1
                        χ′_nλ1 = χ′_nλα[n, λ1, α]
                        Γ′_nλ1 = Γ′_nλα[n, λ1, α]

                        for λ1_p in 1:N_λ1
                            χ_npλ1p = χ_nλα[n_p, λ1_p, α_p]
                            Γ_npλ1p = Γ_nλα[n_p, λ1_p, α_p]

                            term1 = -1im * Γ_npλ1p * dot1_λ1[λ1]
                            term2 = -1im * Γ′_nλ1 * dot2_λ1p[λ1_p]
                            pref3 = -1im * (χ_npλ1p + Δα_p - χ′_nλ1 - Δα)

                            Ω_old = Ω_nλ1α_nλ1α[n, λ1, α, n_p, λ1_p, α_p]
                            dΩ_nλ1α_nλ1α[n, λ1, α, n_p, λ1_p, α_p] = term1 + term2 + pref3 * Ω_old
                        end
                    end

                    @inbounds for λ1 in 1:N_λ1
                        χ′_nλ1 = χ′_nλα[n, λ1, α]
                        Γ′_nλ1 = Γ′_nλα[n, λ1, α]

                        for λ2_p in 1:N_λ2
                            λglob_2p = N_λ1 + λ2_p
                            χ_npλ2p = χ_nλα[n_p, λglob_2p, α_p]
                            Γ_npλ2p = Γ_nλα[n_p, λglob_2p, α_p]

                            term1 = -1im * Γ_npλ2p * dot1_λ1[λ1]
                            term2 = -1im * Γ′_nλ1 * dot4_λ2p[λ2_p]
                            pref3 = -1im * (χ_npλ2p + Δα_p - χ′_nλ1 - Δα)

                            Ω_old = Ω_nλ1α_nλ2α[n, λ1, α, n_p, λ2_p, α_p]
                            dΩ_nλ1α_nλ2α[n, λ1, α, n_p, λ2_p, α_p] = term1 + term2 + pref3 * Ω_old
                        end
                    end

                    @inbounds for λ2 in 1:N_λ2
                        λglob_2 = N_λ1 + λ2
                        χ′_nλ2 = χ′_nλα[n, λglob_2, α]
                        Γ′_nλ2 = Γ′_nλα[n, λglob_2, α]

                        for λ1_p in 1:N_λ1
                            χ_npλ1p = χ_nλα[n_p, λ1_p, α_p]
                            Γ_npλ1p = Γ_nλα[n_p, λ1_p, α_p]

                            term1 = -1im * Γ_npλ1p * dot3_λ2[λ2]
                            term2 = -1im * Γ′_nλ2 * dot2_λ1p[λ1_p]
                            pref3 = -1im * (χ_npλ1p + Δα_p - χ′_nλ2 - Δα)

                            Ω_old = Ω_nλ2α_nλ1α[n, λ2, α, n_p, λ1_p, α_p]
                            dΩ_nλ2α_nλ1α[n, λ2, α, n_p, λ1_p, α_p] = term1 + term2 + pref3 * Ω_old
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function eom_tdnegf!(du::Vector{ComplexF64}, u::Vector{ComplexF64}, p::ModelParamsTDNEGF, t)
    v = pointer(u, p)
    dv = pointer(du, p)

    ρ_ab, Ψ_anλα, Ω_nλ1α_nλ1α, Ω_nλ1α_nλ2α, Ω_nλ2α_nλ1α =
        v.ρ_ab, v.Ψ_anλα, v.Ω_nλ1α_nλ1α, v.Ω_nλ1α_nλ2α, v.Ω_nλ2α_nλ1α
    dρ_ab, dΨ_anλα, dΩ_nλ1α_nλ1α, dΩ_nλ1α_nλ2α, dΩ_nλ2α_nλ1α =
        dv.ρ_ab, dv.Ψ_anλα, dv.Ω_nλ1α_nλ1α, dv.Ω_nλ1α_nλ2α, dv.Ω_nλ2α_nλ1α

    Ns = p.Ns
    Nc = p.Nc
    Nα = p.Nα
    N_λ = p.N_λ
    N_λ1 = p.N_λ1
    N_λ2 = p.N_λ2

    _accumulate_Ψ_anα!(p.Ψ_anα, Ψ_anλα, Ns, Nc, N_λ, Nα)
    _build_Π_abα!(p.Π_abα, p.Ψ_anα, p.ξ_anα, Nα)
    _sum_Π_ab!(p.Π_ab, p.Π_abα, Ns, Nα)
    _rhs_ρ!(dρ_ab, ρ_ab, p.H_ab, p.Hρ, p.Π_ab, Ns)

    _prepare_Ψ_products!(p.HΨ_anλα, p.ρξ_anα, p.H_ab, Ψ_anλα, ρ_ab, p.ξ_anα, Nc, N_λ, Nα, Ns)
    _rhs_Ψ_local!(dΨ_anλα, Ψ_anλα, p.HΨ_anλα, p.ξ_anα, p.ρξ_anα, p.χ′_nλα, p.Σᴸ′_nλα, p.Γ′_nλα, p.Δ_α, Ns, Nc, N_λ, Nα)
    _rhs_Ψ_Ω!(dΨ_anλα, Ω_nλ1α_nλ1α, Ω_nλ1α_nλ2α, Ω_nλ2α_nλ1α, p.ξ_anα, p.tmp_Ψ_vec, Ns, Nc, N_λ1, N_λ2, Nα)

    _rhs_Ω!(
        dΩ_nλ1α_nλ1α,
        dΩ_nλ1α_nλ2α,
        dΩ_nλ2α_nλ1α,
        Ω_nλ1α_nλ1α,
        Ω_nλ1α_nλ2α,
        Ω_nλ2α_nλ1α,
        Ψ_anλα,
        p.ξ_anα,
        p.χ_nλα,
        p.Γ_nλα,
        p.χ′_nλα,
        p.Γ′_nλα,
        p.Δ_α,
        p.tmp_λ1,
        p.tmp_λ1p,
        p.tmp_λ2,
        p.tmp_λ2p,
        Nc,
        N_λ1,
        N_λ2,
        Nα,
    )

    return nothing
end

"""
    ExtraBathSpec

Runtime descriptor for extra baths coupled to arbitrary system DOFs.
Currently implemented type: `:nonhermitian_retarded`.
"""
struct ExtraBathSpec
    bath_type::Symbol
    label::Symbol
    idx_couple::Vector{Int}
    ΣR::Matrix{ComplexF64}
    enabled::Bool

    function ExtraBathSpec(
        bath_type::Symbol,
        label::Symbol,
        idx_couple::Vector{Int},
        ΣR::Matrix{ComplexF64},
        enabled::Bool=true,
    )
        length(idx_couple) > 0 || throw(ArgumentError("idx_couple must be non-empty"))
        size(ΣR) == (length(idx_couple), length(idx_couple)) ||
            throw(ArgumentError("size(ΣR) must match length(idx_couple)"))
        return new(bath_type, label, idx_couple, ΣR, enabled)
    end
end

@inline function _build_Γ_full(Ns::Int, bath::ExtraBathSpec)
    Γ_local = 1im .* (bath.ΣR .- bath.ΣR')
    Γ_full = zeros(ComplexF64, Ns, Ns)
    idx = bath.idx_couple
    @inbounds for ia in eachindex(idx), ib in eachindex(idx)
        Γ_full[idx[ia], idx[ib]] = Γ_local[ia, ib]
    end
    return Γ_full
end

@inline function _rhs_ρ_extra_baths!(dρ_ab, ρ_ab, extra_baths::Vector{ExtraBathSpec}, Ns::Int)
    for bath in extra_baths
        bath.enabled || continue
        bath.bath_type == :nonhermitian_retarded ||
            throw(ArgumentError("Unsupported extra bath type $(bath.bath_type) in first implementation"))
        all(1 <= i <= Ns for i in bath.idx_couple) ||
            throw(ArgumentError("Bath $(bath.label) has idx_couple outside 1:Ns"))

        Γ_full = _build_Γ_full(Ns, bath)
        dρ_ab .+= -0.5 .* (Γ_full * ρ_ab .+ ρ_ab * Γ_full)
    end
    return nothing
end

"""
    ExperimentalBlockRHSParams

Preallocated workspaces and immutable metadata for `eom_tdnegf_blocks!`.

The block-based path keeps the flattened state layout (`aux_layout`) separate
from numerical caches:
- per-block reductions/work arrays (`Ψ_an`, `HΨ`, `ρξ`),
- conjugated/self-energy coefficients (`χ′`, `Σᴸ′`, `Γ`, `Γ′`),
- dynamic problem-level ingredients (`H_ab`, `Δ_blocks`),
- Ω-sector workspaces: overlaps `ξΨ[j, i] = ξ_j' Ψ_i` and λ′-sums `C_λ1`, `C_λ2`
  used for the Ψ(Ω) feedback; `thread_min_Ω` sets the Ω-sector size above
  which the elementwise Ω update is threaded.

These caches mirror repeated operations in the legacy solver while avoiding
per-RHS allocations.
"""
Base.@kwdef struct ExperimentalBlockRHSParams
    H_ab::Matrix{ComplexF64}
    # Dynamic block energy shifts (Δ) used in Ψ/Ω RHS terms.
    # Kept at problem level so scans can update Δ without rebuilding blocks.
    Δ_blocks::Vector{ComplexF64}
    # Runtime-only extra baths acting on arbitrary device DOFs.
    extra_baths::Vector{ExtraBathSpec} = ExtraBathSpec[]
    dims_ρ_ab::NTuple{2,Int}
    size_ρ_ab::Int
    aux_layout::SelfEnergyAuxLayout
    block_ranges_Ψ::Vector{UnitRange{Int}}
    pair_ranges_Ω11::Matrix{UnitRange{Int}}
    pair_ranges_Ω12::Matrix{UnitRange{Int}}
    pair_ranges_Ω21::Matrix{UnitRange{Int}}
    blocks::Vector{SelfEnergyBlock}
    Hρ::Matrix{ComplexF64}
    Π_ab::Matrix{ComplexF64}
    Π_abα_obs::Array{ComplexF64,3}
    Ψ_an::Vector{Matrix{ComplexF64}}
    HΨ::Vector{Array{ComplexF64,3}}
    ρξ::Vector{Matrix{ComplexF64}}
    χ′::Vector{Matrix{ComplexF64}}
    Σᴸ′::Vector{Matrix{ComplexF64}}
    Γ::Vector{Matrix{ComplexF64}}
    Γ′::Vector{Matrix{ComplexF64}}
    ξΨ::Matrix{Matrix{ComplexF64}}
    C_λ1::Matrix{Matrix{ComplexF64}}
    C_λ2::Matrix{Matrix{ComplexF64}}
    thread_min_Ω::Int = 1 << 16
    obs_N_sites::Int
    obs_N_loc::Int
    obs_σ_x::SMatrix{2,2,Float64,4}
    obs_σ_y::SMatrix{2,2,ComplexF64,4}
    obs_σ_z::SMatrix{2,2,Float64,4}
    obs_site_ranges::Vector{UnitRange{Int}}
end

function ExperimentalBlockRHSParams(
    H_ab::Matrix{ComplexF64},
    blocks::Vector{SelfEnergyBlock},
    Δ_blocks::Vector{ComplexF64},
    extra_baths::Vector{ExtraBathSpec}=ExtraBathSpec[],
)
    Ns = size(H_ab, 1)
    size(H_ab, 2) == Ns || throw(ArgumentError("H_ab must be square"))
    length(Δ_blocks) == length(blocks) || throw(ArgumentError("length(Δ_blocks) must match length(blocks)"))
    dims_ρ_ab = (Ns, Ns)
    size_ρ_ab = Ns * Ns
    aux_layout = build_selfenergy_aux_layout(blocks)

    nblocks = length(blocks)
    block_ranges_Ψ = Vector{UnitRange{Int}}(undef, nblocks)
    pair_ranges_Ω11 = Matrix{UnitRange{Int}}(undef, nblocks, nblocks)
    pair_ranges_Ω12 = Matrix{UnitRange{Int}}(undef, nblocks, nblocks)
    pair_ranges_Ω21 = Matrix{UnitRange{Int}}(undef, nblocks, nblocks)

    for i in 1:nblocks
        block_ranges_Ψ[i] = (first(aux_layout.block_layouts[i].range_Ψ) + size_ρ_ab):(last(aux_layout.block_layouts[i].range_Ψ) + size_ρ_ab)
        for j in 1:nblocks
            pl = aux_layout.pair_layouts[i, j]
            pair_ranges_Ω11[i, j] = (first(pl.range_Ω11) + size_ρ_ab):(last(pl.range_Ω11) + size_ρ_ab)
            pair_ranges_Ω12[i, j] = (first(pl.range_Ω12) + size_ρ_ab):(last(pl.range_Ω12) + size_ρ_ab)
            pair_ranges_Ω21[i, j] = (first(pl.range_Ω21) + size_ρ_ab):(last(pl.range_Ω21) + size_ρ_ab)
        end
    end

    for (i, block) in enumerate(blocks)
        block_layout = aux_layout.block_layouts[i]
        size(block.ξ_an, 1) == Ns || throw(ArgumentError("block $(block.name) has incompatible ξ_an row-size; expected $Ns, got $(size(block.ξ_an, 1))"))
        block.N_λ == block.N_λ1 + block.N_λ2 || throw(ArgumentError("block $(block.name) has inconsistent λ split: N_λ=$(block.N_λ), N_λ1+N_λ2=$(block.N_λ1 + block.N_λ2)"))
        block_layout.Nc == block.Nc || throw(ArgumentError("layout/block Nc mismatch for block $(block.name)"))
        block_layout.N_λ1 == block.N_λ1 || throw(ArgumentError("layout/block N_λ1 mismatch for block $(block.name)"))
        block_layout.N_λ2 == block.N_λ2 || throw(ArgumentError("layout/block N_λ2 mismatch for block $(block.name)"))
        block_layout.N_λ == block.N_λ || throw(ArgumentError("layout/block N_λ mismatch for block $(block.name)"))
    end

    Ψ_an = [zeros(ComplexF64, Ns, b.Nc) for b in blocks]
    HΨ = [zeros(ComplexF64, Ns, b.Nc, b.N_λ) for b in blocks]
    ρξ = [zeros(ComplexF64, Ns, b.Nc) for b in blocks]

    # Keep Γ convention aligned with the legacy path: Γ = 1im * (Σᴳ - Σᴸ).
    Γ = [1im .* (b.ΣG_nλ .- b.ΣL_nλ) for b in blocks]
    χ′ = [conj.(b.χ_nλ) for b in blocks]
    Σᴸ′ = [conj.(b.ΣL_nλ) for b in blocks]
    Γ′ = [conj.(Γi) for Γi in Γ]

    return ExperimentalBlockRHSParams(
        H_ab = H_ab,
        Δ_blocks = Δ_blocks,
        extra_baths = extra_baths,
        dims_ρ_ab = dims_ρ_ab,
        size_ρ_ab = size_ρ_ab,
        aux_layout = aux_layout,
        block_ranges_Ψ = block_ranges_Ψ,
        pair_ranges_Ω11 = pair_ranges_Ω11,
        pair_ranges_Ω12 = pair_ranges_Ω12,
        pair_ranges_Ω21 = pair_ranges_Ω21,
        blocks = blocks,
        Hρ = zeros(ComplexF64, Ns, Ns),
        Π_ab = zeros(ComplexF64, Ns, Ns),
        Π_abα_obs = zeros(ComplexF64, Ns, Ns, length(blocks)),
        Ψ_an = Ψ_an,
        HΨ = HΨ,
        ρξ = ρξ,
        χ′ = χ′,
        Σᴸ′ = Σᴸ′,
        Γ = Γ,
        Γ′ = Γ′,
        ξΨ = [zeros(ComplexF64, blocks[j].Nc, blocks[i].Nc * blocks[i].N_λ) for j in 1:nblocks, i in 1:nblocks],
        C_λ1 = [zeros(ComplexF64, blocks[i].Nc * blocks[i].N_λ1, blocks[j].Nc) for i in 1:nblocks, j in 1:nblocks],
        C_λ2 = [zeros(ComplexF64, blocks[i].Nc * blocks[i].N_λ2, blocks[j].Nc) for i in 1:nblocks, j in 1:nblocks],
        obs_N_sites = 0,
        obs_N_loc = 0,
        obs_σ_x = SMatrix{2,2,Float64,4}(0.0, 1.0, 1.0, 0.0),
        obs_σ_y = SMatrix{2,2,ComplexF64,4}(0.0 + 0.0im, 1im, -1im, 0.0 + 0.0im),
        obs_σ_z = SMatrix{2,2,Float64,4}(1.0, 0.0, 0.0, -1.0),
        obs_site_ranges = UnitRange{Int}[]
    )
end

function ExperimentalBlockRHSParams(
    H_ab::Matrix{ComplexF64},
    blocks::Vector{SelfEnergyBlock},
    Δ_blocks::Vector{ComplexF64},
    p_obs::ModelParamsTDNEGF,
    extra_baths::Vector{ExtraBathSpec}=ExtraBathSpec[],
)
    # Preferred constructor for block observables: it stores geometry and
    # Pauli matrices required by obs_Ixα!(ptr, p_blocks, obs).
    p = ExperimentalBlockRHSParams(H_ab, blocks, Δ_blocks, extra_baths)
    obs_N_sites = p_obs.N_sites
    obs_N_loc = p_obs.N_loc
    obs_N_sites * obs_N_loc == p.dims_ρ_ab[1] || throw(ArgumentError("Observable geometry mismatch: N_sites*N_loc = $(obs_N_sites * obs_N_loc), expected $(p.dims_ρ_ab[1])."))
    obs_site_ranges = [get_sub(i, obs_N_loc) for i in 1:obs_N_sites]
    return ExperimentalBlockRHSParams(
        H_ab = p.H_ab,
        Δ_blocks = p.Δ_blocks,
        extra_baths = p.extra_baths,
        dims_ρ_ab = p.dims_ρ_ab,
        size_ρ_ab = p.size_ρ_ab,
        aux_layout = p.aux_layout,
        block_ranges_Ψ = p.block_ranges_Ψ,
        pair_ranges_Ω11 = p.pair_ranges_Ω11,
        pair_ranges_Ω12 = p.pair_ranges_Ω12,
        pair_ranges_Ω21 = p.pair_ranges_Ω21,
        blocks = p.blocks,
        Hρ = p.Hρ,
        Π_ab = p.Π_ab,
        Π_abα_obs = p.Π_abα_obs,
        Ψ_an = p.Ψ_an,
        HΨ = p.HΨ,
        ρξ = p.ρξ,
        χ′ = p.χ′,
        Σᴸ′ = p.Σᴸ′,
        Γ = p.Γ,
        Γ′ = p.Γ′,
        ξΨ = p.ξΨ,
        C_λ1 = p.C_λ1,
        C_λ2 = p.C_λ2,
        thread_min_Ω = p.thread_min_Ω,
        obs_N_sites = obs_N_sites,
        obs_N_loc = obs_N_loc,
        obs_σ_x = p_obs.σ_x,
        obs_σ_y = p_obs.σ_y,
        obs_σ_z = p_obs.σ_z,
        obs_site_ranges = obs_site_ranges,
    )
end

"""
    update_H_e!(p::ExperimentalBlockRHSParams, H0, site_ranges, S, j_sd; Ny, σx, σy, σz)

In-place Hamiltonian update for the block backend. This mirrors the
`ModelParamsTDNEGF` update rule while keeping `H_ab` as a dynamic
problem-level object alongside `Δ_blocks`.
"""
@inline function update_H_e!(p::ExperimentalBlockRHSParams, H0::Matrix{ComplexF64},
                            site_ranges::Vector{UnitRange{Int64}},
                            S::Matrix{SVector{3,Float64}},
                            j_sd::Float64;
                            Ny::Int,
                            σx::SMatrix{2,2,Float64,4}=SMatrix{2,2,Float64,4}(0.0, 1.0, 1.0, 0.0),
                            σy::SMatrix{2,2,ComplexF64,4}=SMatrix{2,2,ComplexF64,4}(0.0 + 0.0im, 1im, -1im, 0.0 + 0.0im),
                            σz::SMatrix{2,2,Float64,4}=SMatrix{2,2,Float64,4}(1.0, 0.0, 0.0, -1.0))
    H = p.H_ab
    @assert size(H, 1) == size(H0, 1) == size(H0, 2) "H/H0 dimensions must match"
    @inbounds for l in 1:length(site_ranges)
        a, b = ij_from_linear(l, Ny)
        r = site_ranges[l]
        Sx = S[a,b][1]; Sy = S[a,b][2]; Sz = S[a,b][3]
        H[r[1], r[1]] = H0[r[1], r[1]] - j_sd*(Sx*σx[1,1] + Sy*σy[1,1] + Sz*σz[1,1])
        H[r[1], r[2]] = H0[r[1], r[2]] - j_sd*(Sx*σx[1,2] + Sy*σy[1,2] + Sz*σz[1,2])
        H[r[2], r[1]] = H0[r[2], r[1]] - j_sd*(Sx*σx[2,1] + Sy*σy[2,1] + Sz*σz[2,1])
        H[r[2], r[2]] = H0[r[2], r[2]] - j_sd*(Sx*σx[2,2] + Sy*σy[2,2] + Sz*σz[2,2])
    end
    return nothing
end

# dΩ[n, λ, n′, λ′] = -iΓ_j[n′,λ′] ξΨ_ji[n′,(n,λ)] - iΓ′_i[n,λ] conj(ξΨ_ij[n,(n′,λ′)])
#                    - i(χ_j[n′,λ′] + Δ_j - χ′_i[n,λ] - Δ_i) Ω[n, λ, n′, λ′]
# λ ranges over `λs_i` (global index into block i), λ′ over `λs_j` (block j).
@inline function _omega_column!(du, u, off, lp, Nc_i, Nc_j, λs_i, λs_j,
                                ξΨ_ji, ξΨ_ij, Γ_j, Γ′_i, χ_j, χ′_i, Δ_i, Δ_j)
    # Ω sector stored contiguously at u[off + 1 : ...] as [n, l, n′, l′].
    λg_p = λs_j[lp]
    k = off + (lp - 1) * Nc_i * length(λs_i) * Nc_j
    @inbounds for n_p in 1:Nc_j
        Γjp = Γ_j[n_p, λg_p]
        cj = χ_j[n_p, λg_p] + Δ_j
        col_ij = n_p + Nc_j * (λg_p - 1)
        for λg in λs_i
            col_ji0 = Nc_i * (λg - 1)
            @simd for n in 1:Nc_i
                du[k + n] = -1im * Γjp * ξΨ_ji[n_p, n + col_ji0] -
                            1im * Γ′_i[n, λg] * conj(ξΨ_ij[n, col_ij]) -
                            1im * (cj - χ′_i[n, λg] - Δ_i) * u[k + n]
            end
            k += Nc_i
        end
    end
    return nothing
end

function _omega_sector!(du, u, range, Nc_i, Nc_j, λs_i, λs_j, ξΨ_ji, ξΨ_ij, Γ_j, Γ′_i, χ_j, χ′_i, Δ_i, Δ_j, thread_min)
    off = first(range) - 1
    nlp = length(λs_j)
    if Threads.nthreads() > 1 && length(range) >= thread_min
        Threads.@threads :static for lp in 1:nlp
            _omega_column!(du, u, off, lp, Nc_i, Nc_j, λs_i, λs_j, ξΨ_ji, ξΨ_ij, Γ_j, Γ′_i, χ_j, χ′_i, Δ_i, Δ_j)
        end
    else
        for lp in 1:nlp
            _omega_column!(du, u, off, lp, Nc_i, Nc_j, λs_i, λs_j, ξΨ_ji, ξΨ_ij, Γ_j, Γ′_i, χ_j, χ′_i, Δ_i, Δ_j)
        end
    end
    return nothing
end

# c[k] (+)= Σ_l u[off + k + m(l-1)] for k in 1:m, l in 1:nl: sum over the last index
# of a contiguous m × nl block (faster than gemv on a reshaped view).
@inline function _sum_last!(c, u, range, m, nl, accumulate::Bool)
    off = first(range) - 1
    accumulate || fill!(c, 0.0 + 0.0im)
    @inbounds for l in 1:nl
        k0 = off + m * (l - 1)
        @simd for k in 1:m
            c[k] += u[k0 + k]
        end
    end
    return nothing
end

"""
    eom_tdnegf_blocks!(du, u, p, t)

In-place RHS for the experimental heterogeneous block-based auxiliary solver.

State layout matches `pointer_blocks`:
`u = [vec(ρ_ab); vec(Ψ blocks); vec(Ω pair sectors)]`.
The equations are the same as the legacy ξ-based path, but applied per block
and per ordered block pair `(i, j)`.

Ω sectors are traversed in memory order (first index innermost) using flat
indices into `u`/`du`; the overlaps ξ_j' Ψ_i are computed with one BLAS product
per block pair. The RHS does not allocate; sectors with at least
`p.thread_min_Ω` elements are updated with `Threads.@threads`.
"""
function eom_tdnegf_blocks!(du::Vector{ComplexF64}, u::Vector{ComplexF64}, p::ExperimentalBlockRHSParams, t)
    ρ_ab = reshape(view(u, 1:p.size_ρ_ab), p.dims_ρ_ab)
    dρ_ab = reshape(view(du, 1:p.size_ρ_ab), p.dims_ρ_ab)
    Ns = p.dims_ρ_ab[1]
    nb = length(p.blocks)

    fill!(p.Π_ab, 0.0 + 0.0im)

    # Overlaps ξ_j' Ψ_i for every ordered pair, one gemm each.
    for i in 1:nb
        Ψ_i_mat = reshape(view(u, p.block_ranges_Ψ[i]), Ns, :)
        for j in 1:nb
            mul!(p.ξΨ[j, i], adjoint(p.blocks[j].ξ_an), Ψ_i_mat)
        end
    end

    @inbounds for i in 1:nb
        block_i = p.blocks[i]
        Δ_i = p.Δ_blocks[i]
        Nc_i, N_λ1_i, N_λ2_i, N_λ_i = block_i.Nc, block_i.N_λ1, block_i.N_λ2, block_i.N_λ

        Ψ_i = reshape(view(u, p.block_ranges_Ψ[i]), (Ns, Nc_i, N_λ_i))
        dΨ_i = reshape(view(du, p.block_ranges_Ψ[i]), (Ns, Nc_i, N_λ_i))

        Ψ_an_i = p.Ψ_an[i]
        HΨ_i = p.HΨ[i]
        ρξ_i = p.ρξ[i]
        χ′_i = p.χ′[i]
        Σᴸ′_i = p.Σᴸ′[i]
        Γ′_i = p.Γ′[i]

        for n in 1:Nc_i, a in 1:Ns
            acc = 0.0 + 0.0im
            @simd for λ in 1:N_λ_i
                acc += Ψ_i[a, n, λ]
            end
            Ψ_an_i[a, n] = acc
        end

        mul!(reshape(HΨ_i, Ns, Nc_i * N_λ_i), p.H_ab, reshape(Ψ_i, Ns, Nc_i * N_λ_i))
        mul!(ρξ_i, ρ_ab, block_i.ξ_an)
        mul!(p.Π_ab, Ψ_an_i, transpose(block_i.ξ_an), 1.0 + 0.0im, 1.0 + 0.0im)

        for λ in 1:N_λ_i, n in 1:Nc_i
            coef_χΨ = 1im * (χ′_i[n, λ] + Δ_i)
            coef_Σξ = 1im * Σᴸ′_i[n, λ]
            coef_Γρξ = -Γ′_i[n, λ]
            @simd for a in 1:Ns
                dΨ_i[a, n, λ] = -1im * HΨ_i[a, n, λ] + coef_χΨ * Ψ_i[a, n, λ] +
                                coef_Σξ * block_i.ξ_an[a, n] + coef_Γρξ * ρξ_i[a, n]
            end
        end

        # Ψ(Ω) feedback: dΨ_i[:, (n,λ)] += -i Σ_j Σ_n′ C[(n,λ), n′] ξ_j[:, n′]
        r = p.block_ranges_Ψ[i]
        dΨ1 = reshape(view(du, first(r):(first(r) + Ns * Nc_i * N_λ1_i - 1)), Ns, Nc_i * N_λ1_i)
        dΨ2 = reshape(view(du, (first(r) + Ns * Nc_i * N_λ1_i):last(r)), Ns, Nc_i * N_λ2_i)
        for j in 1:nb
            block_j = p.blocks[j]
            Nc_j, N_λ1_j, N_λ2_j = block_j.Nc, block_j.N_λ1, block_j.N_λ2
            m1 = Nc_i * N_λ1_i * Nc_j
            m2 = Nc_i * N_λ2_i * Nc_j
            _sum_last!(p.C_λ1[i, j], u, p.pair_ranges_Ω11[i, j], m1, N_λ1_j, false)
            _sum_last!(p.C_λ1[i, j], u, p.pair_ranges_Ω12[i, j], m1, N_λ2_j, true)
            _sum_last!(p.C_λ2[i, j], u, p.pair_ranges_Ω21[i, j], m2, N_λ1_j, false)
            mul!(dΨ1, block_j.ξ_an, transpose(p.C_λ1[i, j]), -1.0im, 1.0 + 0.0im)
            mul!(dΨ2, block_j.ξ_an, transpose(p.C_λ2[i, j]), -1.0im, 1.0 + 0.0im)
        end
    end

    @inbounds for i in 1:nb, j in 1:nb
        block_i, block_j = p.blocks[i], p.blocks[j]
        Δ_i, Δ_j = p.Δ_blocks[i], p.Δ_blocks[j]
        Nc_i, N_λ1_i, N_λ2_i = block_i.Nc, block_i.N_λ1, block_i.N_λ2
        Nc_j, N_λ1_j, N_λ2_j = block_j.Nc, block_j.N_λ1, block_j.N_λ2
        λ1_i, λ2_i = 1:N_λ1_i, (N_λ1_i + 1):(N_λ1_i + N_λ2_i)
        λ1_j, λ2_j = 1:N_λ1_j, (N_λ1_j + 1):(N_λ1_j + N_λ2_j)
        args = (p.ξΨ[j, i], p.ξΨ[i, j], p.Γ[j], p.Γ′[i], block_j.χ_nλ, p.χ′[i], Δ_i, Δ_j, p.thread_min_Ω)

        _omega_sector!(du, u, p.pair_ranges_Ω11[i, j], Nc_i, Nc_j, λ1_i, λ1_j, args...)
        _omega_sector!(du, u, p.pair_ranges_Ω12[i, j], Nc_i, Nc_j, λ1_i, λ2_j, args...)
        _omega_sector!(du, u, p.pair_ranges_Ω21[i, j], Nc_i, Nc_j, λ2_i, λ1_j, args...)
    end

    _rhs_ρ!(dρ_ab, ρ_ab, p.H_ab, p.Hρ, p.Π_ab, Ns)
    _rhs_ρ_extra_baths!(dρ_ab, ρ_ab, p.extra_baths, Ns)
    return nothing
end
