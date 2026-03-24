using Test
using TDNEGF

function make_block(name::Symbol, Ns::Int, Nc::Int, N_λ1::Int, N_λ2::Int)
    N_λ = N_λ1 + N_λ2
    ΣL_nλ = zeros(ComplexF64, Nc, N_λ)
    ΣG_nλ = zeros(ComplexF64, Nc, N_λ)
    χ_nλ = zeros(ComplexF64, Nc, N_λ)
    ξ_an = zeros(ComplexF64, Ns, Nc)
    Δ = 0.0 + 0.0im
    return SelfEnergyBlock(name, Nc, N_λ1, N_λ2, N_λ, ΣL_nλ, ΣG_nλ, χ_nλ, ξ_an, Δ)
end

@testset "Self-energy auxiliary block layout" begin
    block1 = make_block(:left, 8, 2, 3, 1)
    block2 = make_block(:right, 8, 3, 2, 2)

    layouts, total_size = build_selfenergy_aux_layout([block1, block2])

    @test length(layouts) == 2

    l1 = layouts[1]
    @test l1.size_Ψ == 8 * 2 * 4
    @test l1.size_Ω11 == 2 * 3 * 2 * 3
    @test l1.size_Ω12 == 2 * 3 * 2 * 1
    @test l1.size_Ω21 == 2 * 1 * 2 * 3
    @test l1.offset == 1
    @test first(l1.range_block) == 1
    @test last(l1.range_block) == l1.size_block

    l2 = layouts[2]
    @test l2.size_Ψ == 8 * 3 * 4
    @test l2.size_Ω11 == 3 * 2 * 3 * 2
    @test l2.size_Ω12 == 3 * 2 * 3 * 2
    @test l2.size_Ω21 == 3 * 2 * 3 * 2
    @test l2.offset == l1.size_block + 1
    @test first(l2.range_block) == l1.size_block + 1
    @test last(l2.range_block) == l1.size_block + l2.size_block

    @test total_size == l1.size_block + l2.size_block
end
