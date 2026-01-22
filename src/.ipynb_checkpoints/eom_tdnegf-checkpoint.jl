### code for the differential equations using MPM and pade decomposition 
function eom_tdnegf!(du::Vector{ComplexF64}, u::Vector{ComplexF64}, p::ModelParamsTDNEGF, t)
    ####----------------------------------------------------
    #### 0. Pointer al vector de estado y desempaquetado
    ####----------------------------------------------------
    v  = pointer(u,  p)
    dv = pointer(du, p)

    # Variables dinámicas (estado y derivadas)
    ρ_ab, Ψ_anλα, Ω_nλ1α_nλ1α, Ω_nλ1α_nλ2α, Ω_nλ2α_nλ1α      =
        v.ρ_ab,  v.Ψ_anλα,  v.Ω_nλ1α_nλ1α,  v.Ω_nλ1α_nλ2α,  v.Ω_nλ2α_nλ1α
    dρ_ab, dΨ_anλα, dΩ_nλ1α_nλ1α, dΩ_nλ1α_nλ2α, dΩ_nλ2α_nλ1α =
        dv.ρ_ab, dv.Ψ_anλα, dv.Ω_nλ1α_nλ1α, dv.Ω_nλ1α_nλ2α, dv.Ω_nλ2α_nλ1α

    ####----------------------------------------------------
    #### 1. Alias de tamaños y parámetros pre-calculados
    ####----------------------------------------------------
    Ns   = p.Ns
    Nc   = p.Nc
    Nα   = p.Nα
    N_λ  = p.N_λ
    N_λ1 = p.N_λ1
    N_λ2 = p.N_λ2

    Σᴸ_nλα = p.Σᴸ_nλα
    Σᴳ_nλα = p.Σᴳ_nλα
    Γ_nλα  = p.Γ_nλα          # Γ    (sin conjugar)
    χ_nλα  = p.χ_nλα          # χ    (sin conjugar)

    # Conjugados precomputados (primados)
    χ′_nλα  = p.χ′_nλα        # χ′ = conj.(χ)
    Σᴸ′_nλα = p.Σᴸ′_nλα       # Σᴸ′ = conj.(Σᴸ)
    Γ′_nλα  = p.Γ′_nλα        # Γ′  = conj.(Γ)

    ξ_anα   = p.ξ_anα
    Δ_α     = p.Δ_α
    H_ab    = p.H_ab
    Π_abα   = p.Π_abα

    # Buffers de trabajo
    Hρ      = p.Hρ            # Ns × Ns
    Ψ_anα   = p.Ψ_anα         # Ns × Nc × Nα
    Π_ab    = p.Π_ab          # Ns × Ns

    HΨ_anλα = p.HΨ_anλα       # Ns × Nc × N_λ × Nα
    ρξ_anα  = p.ρξ_anα        # Ns × Nc × Nα

    tmp_Ψ_a = p.tmp_Ψ_vec     # Ns

    # Buffers escalares en espacio de polos
    dot1_λ1  = p.tmp_λ1       # N_λ1
    dot2_λ1p = p.tmp_λ1p      # N_λ1
    dot3_λ2  = p.tmp_λ2       # N_λ2
    dot4_λ2p = p.tmp_λ2p      # N_λ2

    ####====================================================
    #### BLOQUE ρ: dρ_ab
    ####====================================================

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

    # 1B) Π_abα(⋅,⋅,α) = Ψ_anα(⋅,⋅,α) * ξ_anα(⋅,⋅,α)^T
    @inbounds for α in 1:Nα
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

    # 1D) Hρ = H_ab * ρ_ab ; dρ = -i(Hρ - Hρ†) + (Π_ab + Π_ab†)
    mul!(Hρ, H_ab, ρ_ab)
    @inbounds for a in 1:Ns
        @simd for b in 1:Ns
            comm   = Hρ[a, b] - conj(Hρ[b, a])
            source = Π_ab[a, b] + conj(Π_ab[b, a])
            dρ_ab[a, b] = -1im * comm + source
        end
    end

    ####====================================================
    #### BLOQUE Ψ (sin Ω): H, χ, Σ, Γ
    ####   dΨ = -i HΨ + i(χ′+Δ)Ψ + i Σᴸ′ ξ - Γ′ (ρξ)
    ####====================================================

    # 2A) Reinterpretar como matrices y hacer GEMM grandes
    Ncols_Ψ = Nc * N_λ * Nα
    Ncols_ξ = Nc * Nα

    Ψ_mat   = reshape(Ψ_anλα,    Ns, Ncols_Ψ)
    HΨ_mat  = reshape(HΨ_anλα,   Ns, Ncols_Ψ)
    ξ_mat   = reshape(ξ_anα,     Ns, Ncols_ξ)
    ρξ_mat  = reshape(ρξ_anα,    Ns, Ncols_ξ)

    # HΨ_anλα = H_ab * Ψ_anλα
    mul!(HΨ_mat, H_ab, Ψ_mat)

    # ρξ_anα = ρ_ab * ξ_anα
    mul!(ρξ_mat, ρ_ab, ξ_mat)

    # 2B) dΨ_anλα contribución local en (a,n,λ,α)
    @inbounds for α in 1:Nα
        Δα = Δ_α[α]
        for n in 1:Nc
            ξ_anα_nα  = @view ξ_anα[:,  n, α]
            ρξ_anα_nα = @view ρξ_anα[:, n, α]

            for λ in 1:N_λ
                χ′ = χ′_nλα[n, λ, α]
                Σᴸ′ = Σᴸ′_nλα[n, λ, α]
                Γ′  = Γ′_nλα[n, λ, α]

                coef_χΨ  = 1im * (χ′ + Δα)
                coef_Σξ  = 1im * Σᴸ′
                coef_Γρξ = -Γ′

                Ψ_anλα_nλα  = @view Ψ_anλα[:,  n, λ, α]
                HΨ_anλα_nλα = @view HΨ_anλα[:, n, λ, α]
                dΨ_anλα_nλα = @view dΨ_anλα[:, n, λ, α]

                @simd for a in 1:Ns
                    dΨ_anλα_nλα[a] =
                        -1im * HΨ_anλα_nλα[a] +
                        coef_χΨ  * Ψ_anλα_nλα[a] +
                        coef_Σξ  * ξ_anα_nα[a] +
                        coef_Γρξ * ρξ_anα_nα[a]
                end
            end
        end
    end

    ####====================================================
    #### BLOQUE Ψ (con Ω): dΨ += -i Ω · ξ
    ####   Separado en Ω_nλ1α_nλ1α, Ω_nλ1α_nλ2α, Ω_nλ2α_nλ1α
    ####====================================================

    # 3A) Contribuciones de Ω_nλ1α_nλ1α y Ω_nλ1α_nλ2α  (λ ≤ N_λ1)
    @inbounds for α in 1:Nα
        for n in 1:Nc
            for λ1 in 1:N_λ1
                fill!(tmp_Ψ_a, 0.0 + 0.0im)

                for α_p in 1:Nα
                    for n_p in 1:Nc
                        # coef_total = -i [ Σ_{λ1′} Ω11 + Σ_{λ2′} Ω12 ]
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

    # 3B) Contribuciones de Ω_nλ2α_nλ1α  (λ > N_λ1)
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

    ####====================================================
    #### BLOQUE Ω: ecuaciones para dΩ_nλ1α_nλ1α, dΩ_nλ1α_nλ2α, dΩ_nλ2α_nλ1α
    ####   Aquí sí necesitamos χ y χ′, Γ y Γ′.
    ####====================================================

    @inbounds for α in 1:Nα
        Δα = Δ_α[α]
        for α_p in 1:Nα
            Δα_p = Δ_α[α_p]

            for n in 1:Nc
                ξ_anα_nα     = @view ξ_anα[:, n,  α]
                for n_p in 1:Nc
                    ξ_anα_npαp  = @view ξ_anα[:, n_p, α_p]

                    #### Productos en espacio a que se reutilizan ####

                    # dot1_λ1[λ1] = ⟨ξ_{n′α′} | Ψ_{n,λ1,α}⟩
                    @inbounds for λ1 in 1:N_λ1
                        Ψ_anλα_nλ1α = @view Ψ_anλα[:, n, λ1, α]
                        dot1_λ1[λ1] = dot(ξ_anα_npαp, Ψ_anλα_nλ1α)
                    end

                    # dot2_λ1p[λ1′] = ⟨ξ_{nα} | Ψ_{n′,λ1′,α′}⟩*
                    @inbounds for λ1_p in 1:N_λ1
                        Ψ_anλα_npλ1pαp = @view Ψ_anλα[:, n_p, λ1_p, α_p]
                        dot_tmp        = dot(ξ_anα_nα, Ψ_anλα_npλ1pαp)
                        dot2_λ1p[λ1_p] = conj(dot_tmp)
                    end

                    # dot3_λ2[λ2] = ⟨ξ_{n′α′} | Ψ_{n,λ2+N_λ1,α}⟩
                    @inbounds for λ2 in 1:N_λ2
                        Ψ_anλα_nλ2α = @view Ψ_anλα[:, n, N_λ1 + λ2, α]
                        dot3_λ2[λ2] = dot(ξ_anα_npαp, Ψ_anλα_nλ2α)
                    end

                    # dot4_λ2p[λ2′] = ⟨ξ_{nα} | Ψ_{n′,λ2′+N_λ1,α′}⟩*
                    @inbounds for λ2_p in 1:N_λ2
                        Ψ_anλα_npλ2pαp = @view Ψ_anλα[:, n_p, N_λ1 + λ2_p, α_p]
                        dot_tmp        = dot(ξ_anα_nα, Ψ_anλα_npλ2pαp)
                        dot4_λ2p[λ2_p] = conj(dot_tmp)
                    end

                    #### 4.1) dΩ_nλ1α_nλ1α ####
                    @inbounds for λ1 in 1:N_λ1
                        χ_nλ1   = χ_nλα[n,  λ1,   α ]
                        Γ_nλ1   = Γ_nλα[n,  λ1,   α ]
                        χ′_nλ1  = χ′_nλα[n, λ1,   α ] # conj(χ_nλ1)
                        Γ′_nλ1  = Γ′_nλα[n, λ1,   α ] # conj(Γ_nλ1)

                        for λ1_p in 1:N_λ1
                            χ_npλ1p  = χ_nλα[n_p, λ1_p, α_p]
                            Γ_npλ1p  = Γ_nλα[n_p, λ1_p, α_p]

                            term1 = -1im * Γ_npλ1p * dot1_λ1[λ1]
                            term2 = -1im * Γ′_nλ1   * dot2_λ1p[λ1_p]

                            pref3 = -1im * (χ_npλ1p + Δα_p -
                                            χ′_nλ1      - Δα)

                            Ω_old = Ω_nλ1α_nλ1α[n, λ1, α, n_p, λ1_p, α_p]

                            dΩ_nλ1α_nλ1α[n, λ1, α, n_p, λ1_p, α_p] =
                                term1 + term2 + pref3 * Ω_old
                        end
                    end

                    #### 4.2) dΩ_nλ1α_nλ2α ####
                    @inbounds for λ1 in 1:N_λ1
                        χ_nλ1   = χ_nλα[n, λ1, α]
                        Γ_nλ1   = Γ_nλα[n, λ1, α]
                        χ′_nλ1  = χ′_nλα[n, λ1, α]   # conj(χ_nλ1)
                        Γ′_nλ1  = Γ′_nλα[n, λ1, α]   # conj(Γ_nλ1)

                        for λ2_p in 1:N_λ2
                            λglob_2p = N_λ1 + λ2_p
                            χ_npλ2p  = χ_nλα[n_p, λglob_2p, α_p]
                            Γ_npλ2p  = Γ_nλα[n_p, λglob_2p, α_p]

                            term1 = -1im * Γ_npλ2p * dot1_λ1[λ1]
                            term2 = -1im * Γ′_nλ1  * dot4_λ2p[λ2_p]

                            pref3 = -1im * (χ_npλ2p + Δα_p -
                                            χ′_nλ1      - Δα)

                            Ω_old = Ω_nλ1α_nλ2α[n, λ1, α, n_p, λ2_p, α_p]

                            dΩ_nλ1α_nλ2α[n, λ1, α, n_p, λ2_p, α_p] =
                                term1 + term2 + pref3 * Ω_old
                        end
                    end

                    #### 4.3) dΩ_nλ2α_nλ1α ####
                    @inbounds for λ2 in 1:N_λ2
                        λglob_2 = N_λ1 + λ2
                        χ_nλ2   = χ_nλα[n, λglob_2, α]
                        Γ_nλ2   = Γ_nλα[n, λglob_2, α]
                        χ′_nλ2  = χ′_nλα[n, λglob_2, α]   # conj(χ_nλ2)
                        Γ′_nλ2  = Γ′_nλα[n, λglob_2, α]   # conj(Γ_nλ2)

                        for λ1_p in 1:N_λ1
                            χ_npλ1p = χ_nλα[n_p, λ1_p, α_p]
                            Γ_npλ1p = Γ_nλα[n_p, λ1_p, α_p]

                            term1 = -1im * Γ_npλ1p * dot3_λ2[λ2]
                            term2 = -1im * Γ′_nλ2  * dot2_λ1p[λ1_p]

                            pref3 = -1im * (χ_npλ1p + Δα_p -
                                            χ′_nλ2      - Δα)

                            Ω_old = Ω_nλ2α_nλ1α[n, λ2, α, n_p, λ1_p, α_p]

                            dΩ_nλ2α_nλ1α[n, λ2, α, n_p, λ1_p, α_p] =
                                term1 + term2 + pref3 * Ω_old
                        end
                    end

                end # n_p
            end # n
        end # α_p
    end # α

    return nothing
end
