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

@testset "Self-energy auxiliary pair layout" begin
    block1 = make_block(:left, 8, 2, 3, 1)
    block2 = make_block(:right, 8, 3, 2, 2)

    layout = build_selfenergy_aux_layout([block1, block2])

    @test length(layout.block_layouts) == 2
    @test size(layout.pair_layouts) == (2, 2)

    l1 = layout.block_layouts[1]
    l2 = layout.block_layouts[2]
    @test l1.size_Ψ == 8 * 2 * 4
    @test l2.size_Ψ == 8 * 3 * 4
    @test l1.offset == 1
    @test first(l2.range_Ψ) == last(l1.range_Ψ) + 1

    p12 = layout.pair_layouts[1, 2]
    @test p12.size_Ω11 == 2 * 3 * 3 * 2
    @test p12.size_Ω12 == 2 * 3 * 3 * 2
    @test p12.size_Ω21 == 2 * 1 * 3 * 2

    p21 = layout.pair_layouts[2, 1]
    @test p21.size_Ω11 == 3 * 2 * 2 * 3
    @test p21.size_Ω12 == 3 * 2 * 2 * 1
    @test p21.size_Ω21 == 3 * 2 * 2 * 3

    total_Ψ = l1.size_Ψ + l2.size_Ψ
    total_Ω = sum(pl.size_pair for pl in layout.pair_layouts)
    @test layout.total_size == total_Ψ + total_Ω
end

@testset "Heterogeneous pair pointers include cross-block Ω sectors" begin
    dims_ρ_ab = (4, 4)
    size_ρ_ab = prod(dims_ρ_ab)

    block1 = make_block(:left, 4, 2, 2, 1)
    block2 = make_block(:right, 4, 3, 1, 2)
    layout = build_selfenergy_aux_layout([block1, block2])

    vec = zeros(ComplexF64, size_ρ_ab + layout.total_size)
    ptr = pointer_blocks(vec, dims_ρ_ab, layout)

    @test size(ptr.ρ_ab) == dims_ρ_ab
    @test length(ptr.blocks) == 2
    @test size(ptr.Ω_pairs) == (2, 2)

    @test size(ptr.blocks[1].Ψ_anλ) == (4, 2, 3)
    @test size(ptr.blocks[2].Ψ_anλ) == (4, 3, 3)

    p12 = ptr.Ω_pairs[1, 2]
    @test size(p12.Ω11) == (2, 2, 3, 1)
    @test size(p12.Ω12) == (2, 2, 3, 2)
    @test size(p12.Ω21) == (2, 1, 3, 1)

    p21 = ptr.Ω_pairs[2, 1]
    @test size(p21.Ω11) == (3, 1, 2, 2)
    @test size(p21.Ω12) == (3, 1, 2, 1)
    @test size(p21.Ω21) == (3, 2, 2, 2)

    p12.Ω12[1, 2, 3, 2] = 3.0 + 4.0im
    idx_local = LinearIndices(p12.Ω12)[1, 2, 3, 2]
    idx_global = first(p12.range_Ω12) + idx_local - 1
    @test vec[idx_global] == 3.0 + 4.0im
end
