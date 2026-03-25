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

Base.@kwdef struct ExperimentalBlockRHSParams
    H_ab::Matrix{ComplexF64}
    dims_ρ_ab::NTuple{2,Int}
    aux_layout::SelfEnergyAuxLayout
    blocks::Vector{SelfEnergyBlock}
    Hρ::Matrix{ComplexF64}
    Π_ab::Matrix{ComplexF64}
    Ψ_an::Vector{Matrix{ComplexF64}}
    HΨ::Vector{Array{ComplexF64,3}}
    ρξ::Vector{Matrix{ComplexF64}}
    χ′::Vector{Matrix{ComplexF64}}
    Σᴸ′::Vector{Matrix{ComplexF64}}
    Γ::Vector{Matrix{ComplexF64}}
    Γ′::Vector{Matrix{ComplexF64}}
    tmp_Ψ_vec::Vector{ComplexF64}
    tmp_λ1::Vector{Vector{ComplexF64}}
    tmp_λ1p::Vector{Vector{ComplexF64}}
    tmp_λ2::Vector{Vector{ComplexF64}}
    tmp_λ2p::Vector{Vector{ComplexF64}}
end

function ExperimentalBlockRHSParams(H_ab::Matrix{ComplexF64}, blocks::Vector{SelfEnergyBlock})
    Ns = size(H_ab, 1)
    size(H_ab, 2) == Ns || throw(ArgumentError("H_ab must be square"))
    dims_ρ_ab = (Ns, Ns)
    aux_layout = build_selfenergy_aux_layout(blocks)

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
        dims_ρ_ab = dims_ρ_ab,
        aux_layout = aux_layout,
        blocks = blocks,
        Hρ = zeros(ComplexF64, Ns, Ns),
        Π_ab = zeros(ComplexF64, Ns, Ns),
        Ψ_an = Ψ_an,
        HΨ = HΨ,
        ρξ = ρξ,
        χ′ = χ′,
        Σᴸ′ = Σᴸ′,
        Γ = Γ,
        Γ′ = Γ′,
        tmp_Ψ_vec = zeros(ComplexF64, Ns),
        tmp_λ1 = [zeros(ComplexF64, b.N_λ1) for b in blocks],
        tmp_λ1p = [zeros(ComplexF64, b.N_λ1) for b in blocks],
        tmp_λ2 = [zeros(ComplexF64, b.N_λ2) for b in blocks],
        tmp_λ2p = [zeros(ComplexF64, b.N_λ2) for b in blocks],
    )
end

function eom_tdnegf_blocks!(du::Vector{ComplexF64}, u::Vector{ComplexF64}, p::ExperimentalBlockRHSParams, t)
    ptr = pointer_blocks(u, p.dims_ρ_ab, p.aux_layout)
    dptr = pointer_blocks(du, p.dims_ρ_ab, p.aux_layout)

    ρ_ab = ptr.ρ_ab
    dρ_ab = dptr.ρ_ab

    fill!(p.Π_ab, 0.0 + 0.0im)

    @inbounds for i in eachindex(ptr.blocks)
        bptr_i = ptr.blocks[i]
        dbptr_i = dptr.blocks[i]
        block_i = p.blocks[i]

        Ns = p.aux_layout.block_layouts[i].Ns
        Nc_i = block_i.Nc
        N_λ1_i = block_i.N_λ1
        N_λ2_i = block_i.N_λ2
        N_λ_i = block_i.N_λ

        Ψ_an_i = p.Ψ_an[i]
        HΨ_i = p.HΨ[i]
        ρξ_i = p.ρξ[i]
        χ′_i = p.χ′[i]
        Σᴸ′_i = p.Σᴸ′[i]
        Γ′_i = p.Γ′[i]

        @inbounds for n in 1:Nc_i, a in 1:Ns
            acc = 0.0 + 0.0im
            @simd for λ in 1:N_λ_i
                acc += bptr_i.Ψ_anλ[a, n, λ]
            end
            Ψ_an_i[a, n] = acc
        end

        mul!(reshape(HΨ_i, Ns, Nc_i * N_λ_i), p.H_ab, reshape(bptr_i.Ψ_anλ, Ns, Nc_i * N_λ_i))
        mul!(ρξ_i, ρ_ab, block_i.ξ_an)

        mul!(p.Π_ab, Ψ_an_i, transpose(block_i.ξ_an), 1.0 + 0.0im, 1.0 + 0.0im)

        @inbounds for n in 1:Nc_i
            ξ_n = @view block_i.ξ_an[:, n]
            ρξ_n = @view ρξ_i[:, n]

            for λ in 1:N_λ_i
                χ′ = χ′_i[n, λ]
                Σᴸ′ = Σᴸ′_i[n, λ]
                Γ′ = Γ′_i[n, λ]

                coef_χΨ = 1im * (χ′ + block_i.Δ)
                coef_Σξ = 1im * Σᴸ′
                coef_Γρξ = -Γ′

                dΨ_nλ = @view dbptr_i.Ψ_anλ[:, n, λ]
                Ψ_nλ = @view bptr_i.Ψ_anλ[:, n, λ]
                HΨ_nλ = @view HΨ_i[:, n, λ]

                @simd for a in 1:Ns
                    dΨ_nλ[a] = -1im * HΨ_nλ[a] + coef_χΨ * Ψ_nλ[a] + coef_Σξ * ξ_n[a] + coef_Γρξ * ρξ_n[a]
                end
            end
        end

        @inbounds for n in 1:Nc_i
            for λ1 in 1:N_λ1_i
                fill!(p.tmp_Ψ_vec, 0.0 + 0.0im)

                for j in eachindex(ptr.blocks)
                    pair_ij = ptr.Ω_pairs[i, j]
                    block_j = p.blocks[j]
                    Nc_j = block_j.Nc
                    N_λ1_j = block_j.N_λ1
                    N_λ2_j = block_j.N_λ2

                    for n_p in 1:Nc_j
                        coeff = 0.0 + 0.0im
                        @simd for λ1_p in 1:N_λ1_j
                            coeff += pair_ij.Ω11[n, λ1, n_p, λ1_p]
                        end
                        @simd for λ2_p in 1:N_λ2_j
                            coeff += pair_ij.Ω12[n, λ1, n_p, λ2_p]
                        end
                        coeff *= -1im

                        ξ_np = @view block_j.ξ_an[:, n_p]
                        @simd for a in 1:Ns
                            p.tmp_Ψ_vec[a] += coeff * ξ_np[a]
                        end
                    end
                end

                dΨ = @view dbptr_i.Ψ_anλ[:, n, λ1]
                @simd for a in 1:Ns
                    dΨ[a] += p.tmp_Ψ_vec[a]
                end
            end

            for λ2 in 1:N_λ2_i
                λ = N_λ1_i + λ2
                fill!(p.tmp_Ψ_vec, 0.0 + 0.0im)

                for j in eachindex(ptr.blocks)
                    pair_ij = ptr.Ω_pairs[i, j]
                    block_j = p.blocks[j]
                    Nc_j = block_j.Nc
                    N_λ1_j = block_j.N_λ1

                    for n_p in 1:Nc_j
                        coeff = 0.0 + 0.0im
                        @simd for λ1_p in 1:N_λ1_j
                            coeff += pair_ij.Ω21[n, λ2, n_p, λ1_p]
                        end
                        coeff *= -1im

                        ξ_np = @view block_j.ξ_an[:, n_p]
                        @simd for a in 1:Ns
                            p.tmp_Ψ_vec[a] += coeff * ξ_np[a]
                        end
                    end
                end

                dΨ = @view dbptr_i.Ψ_anλ[:, n, λ]
                @simd for a in 1:Ns
                    dΨ[a] += p.tmp_Ψ_vec[a]
                end
            end
        end
    end

    @inbounds for i in eachindex(ptr.blocks)
        bptr_i = ptr.blocks[i]
        block_i = p.blocks[i]
        Nc_i = block_i.Nc
        N_λ1_i = block_i.N_λ1
        N_λ2_i = block_i.N_λ2
        χ′_i = p.χ′[i]
        Γ′_i = p.Γ′[i]

        dot1 = p.tmp_λ1[i]
        dot3 = p.tmp_λ2[i]

        for j in eachindex(ptr.blocks)
            bptr_j = ptr.blocks[j]
            block_j = p.blocks[j]
            Nc_j = block_j.Nc
            N_λ1_j = block_j.N_λ1
            N_λ2_j = block_j.N_λ2
            χ_j = block_j.χ_nλ
            Γ_j = p.Γ[j]

            dot2 = p.tmp_λ1p[j]
            dot4 = p.tmp_λ2p[j]

            pair_ij = ptr.Ω_pairs[i, j]
            dpair_ij = dptr.Ω_pairs[i, j]

            for n in 1:Nc_i
                ξ_i_n = @view block_i.ξ_an[:, n]
                for n_p in 1:Nc_j
                    ξ_j_np = @view block_j.ξ_an[:, n_p]

                    for λ1 in 1:N_λ1_i
                        dot1[λ1] = dot(ξ_j_np, @view bptr_i.Ψ_anλ[:, n, λ1])
                    end
                    for λ1_p in 1:N_λ1_j
                        dot2[λ1_p] = conj(dot(ξ_i_n, @view bptr_j.Ψ_anλ[:, n_p, λ1_p]))
                    end
                    for λ2 in 1:N_λ2_i
                        dot3[λ2] = dot(ξ_j_np, @view bptr_i.Ψ_anλ[:, n, N_λ1_i + λ2])
                    end
                    for λ2_p in 1:N_λ2_j
                        dot4[λ2_p] = conj(dot(ξ_i_n, @view bptr_j.Ψ_anλ[:, n_p, N_λ1_j + λ2_p]))
                    end

                    for λ1 in 1:N_λ1_i
                        χ′_nλ1 = χ′_i[n, λ1]
                        Γ′_nλ1 = Γ′_i[n, λ1]
                        for λ1_p in 1:N_λ1_j
                            χ_j_npl1p = χ_j[n_p, λ1_p]
                            Γ_j_npl1p = Γ_j[n_p, λ1_p]

                            term1 = -1im * Γ_j_npl1p * dot1[λ1]
                            term2 = -1im * Γ′_nλ1 * dot2[λ1_p]
                            pref3 = -1im * (χ_j_npl1p + block_j.Δ - χ′_nλ1 - block_i.Δ)

                            dpair_ij.Ω11[n, λ1, n_p, λ1_p] = term1 + term2 + pref3 * pair_ij.Ω11[n, λ1, n_p, λ1_p]
                        end
                    end

                    for λ1 in 1:N_λ1_i
                        χ′_nλ1 = χ′_i[n, λ1]
                        Γ′_nλ1 = Γ′_i[n, λ1]
                        for λ2_p in 1:N_λ2_j
                            λglob_2p = N_λ1_j + λ2_p
                            χ_j_npl2p = χ_j[n_p, λglob_2p]
                            Γ_j_npl2p = Γ_j[n_p, λglob_2p]

                            term1 = -1im * Γ_j_npl2p * dot1[λ1]
                            term2 = -1im * Γ′_nλ1 * dot4[λ2_p]
                            pref3 = -1im * (χ_j_npl2p + block_j.Δ - χ′_nλ1 - block_i.Δ)

                            dpair_ij.Ω12[n, λ1, n_p, λ2_p] = term1 + term2 + pref3 * pair_ij.Ω12[n, λ1, n_p, λ2_p]
                        end
                    end

                    for λ2 in 1:N_λ2_i
                        λglob_2 = N_λ1_i + λ2
                        χ′_nλ2 = χ′_i[n, λglob_2]
                        Γ′_nλ2 = Γ′_i[n, λglob_2]
                        for λ1_p in 1:N_λ1_j
                            χ_j_npl1p = χ_j[n_p, λ1_p]
                            Γ_j_npl1p = Γ_j[n_p, λ1_p]

                            term1 = -1im * Γ_j_npl1p * dot3[λ2]
                            term2 = -1im * Γ′_nλ2 * dot2[λ1_p]
                            pref3 = -1im * (χ_j_npl1p + block_j.Δ - χ′_nλ2 - block_i.Δ)

                            dpair_ij.Ω21[n, λ2, n_p, λ1_p] = term1 + term2 + pref3 * pair_ij.Ω21[n, λ2, n_p, λ1_p]
                        end
                    end
                end
            end
        end
    end

    _rhs_ρ!(dρ_ab, ρ_ab, p.H_ab, p.Hρ, p.Π_ab, p.dims_ρ_ab[1])

    return nothing
end
