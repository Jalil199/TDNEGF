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

@testset "Heterogeneous auxiliary pointer blocks" begin
    dims_ρ_ab = (4, 4)
    size_ρ_ab = prod(dims_ρ_ab)

    block1 = make_block(:left, 4, 2, 2, 1)
    block2 = make_block(:right, 4, 3, 1, 2)
    layouts, total_aux_size = build_selfenergy_aux_layout([block1, block2])

    vec = zeros(ComplexF64, size_ρ_ab + total_aux_size)
    ptr = pointer_blocks(vec, dims_ρ_ab, layouts)

    @test size(ptr.ρ_ab) == dims_ρ_ab
    @test length(ptr.blocks) == 2

    b1 = ptr.blocks[1]
    @test size(b1.Ψ_anλ) == (4, 2, 3)
    @test size(b1.Ω11) == (2, 2, 2, 2)
    @test size(b1.Ω12) == (2, 2, 2, 1)
    @test size(b1.Ω21) == (2, 1, 2, 2)
    @test first(b1.range_Ψ) == size_ρ_ab + 1
    @test first(b1.range_Ω11) == last(b1.range_Ψ) + 1
    @test first(b1.range_Ω12) == last(b1.range_Ω11) + 1
    @test first(b1.range_Ω21) == last(b1.range_Ω12) + 1

    b2 = ptr.blocks[2]
    @test size(b2.Ψ_anλ) == (4, 3, 3)
    @test size(b2.Ω11) == (3, 1, 3, 1)
    @test size(b2.Ω12) == (3, 1, 3, 2)
    @test size(b2.Ω21) == (3, 2, 3, 1)
    @test first(b2.range_Ψ) == size_ρ_ab + layouts[2].offset
    @test first(b2.range_Ω11) == last(b2.range_Ψ) + 1
    @test first(b2.range_Ω12) == last(b2.range_Ω11) + 1
    @test first(b2.range_Ω21) == last(b2.range_Ω12) + 1
    @test last(b2.range_Ω21) == length(vec)

    b2.Ω12[1, 1, 2, 2] = 3.0 + 4.0im
    @test vec[first(b2.range_Ω12) + 4] == 3.0 + 4.0im
end
