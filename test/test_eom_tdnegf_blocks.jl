using Test
using TDNEGF

@testset "Experimental block RHS smoke test" begin
    Ns = 2
    H_ab = Matrix{ComplexF64}(I, Ns, Ns)

    function mkblock(name, Nc, Nλ1, Nλ2)
        Nλ = Nλ1 + Nλ2
        ΣL = zeros(ComplexF64, Nc, Nλ)
        ΣG = zeros(ComplexF64, Nc, Nλ)
        χ = zeros(ComplexF64, Nc, Nλ)
        ξ = zeros(ComplexF64, Ns, Nc)
        for n in 1:Nc
            ξ[mod1(n, Ns), n] = 1.0 + 0.0im
        end
        return SelfEnergyBlock(name, Nc, Nλ1, Nλ2, Nλ, ΣL, ΣG, χ, ξ, 0.0 + 0.0im)
    end

    blocks = [mkblock(:left, 1, 1, 1), mkblock(:right, 2, 1, 0)]
    p = ExperimentalBlockRHSParams(H_ab, blocks)

    size_ρ = Ns * Ns
    _, total_aux = build_selfenergy_aux_layout(blocks)
    u = zeros(ComplexF64, size_ρ + total_aux)
    du = similar(u)

    eom_tdnegf_blocks!(du, u, p, 0.0)

    @test length(du) == length(u)
    @test all(isfinite, real.(du))
    @test all(isfinite, imag.(du))
end
