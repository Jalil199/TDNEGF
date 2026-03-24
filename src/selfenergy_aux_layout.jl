"""
Lightweight local layout for one heterogeneous self-energy auxiliary block.

This is intentionally forward-looking and does not alter the current solver state layout.
"""
Base.@kwdef struct SelfEnergyAuxBlockLayout
    name::Symbol
    Ns::Int
    Nc::Int
    N_λ1::Int
    N_λ2::Int
    N_λ::Int
    size_Ψ::Int
    size_Ω11::Int
    size_Ω12::Int
    size_Ω21::Int
    size_block::Int
    offset::Int
    range_Ψ::UnitRange{Int}
    range_Ω11::UnitRange{Int}
    range_Ω12::UnitRange{Int}
    range_Ω21::UnitRange{Int}
    range_block::UnitRange{Int}
end

@inline function SelfEnergyAuxBlockLayout(block::SelfEnergyBlock, offset::Int)
    Ns = size(block.ξ_an, 1)
    Nc = block.Nc
    N_λ1 = block.N_λ1
    N_λ2 = block.N_λ2
    N_λ = block.N_λ

    size_Ψ = Ns * Nc * N_λ
    size_Ω11 = Nc * N_λ1 * Nc * N_λ1
    size_Ω12 = Nc * N_λ1 * Nc * N_λ2
    size_Ω21 = Nc * N_λ2 * Nc * N_λ1
    size_block = size_Ψ + size_Ω11 + size_Ω12 + size_Ω21

    range_Ψ = offset:(offset + size_Ψ - 1)
    range_Ω11 = (last(range_Ψ) + 1):(last(range_Ψ) + size_Ω11)
    range_Ω12 = (last(range_Ω11) + 1):(last(range_Ω11) + size_Ω12)
    range_Ω21 = (last(range_Ω12) + 1):(last(range_Ω12) + size_Ω21)
    range_block = offset:(offset + size_block - 1)

    return SelfEnergyAuxBlockLayout(
        ;
        name = block.name,
        Ns,
        Nc,
        N_λ1,
        N_λ2,
        N_λ,
        size_Ψ,
        size_Ω11,
        size_Ω12,
        size_Ω21,
        size_block,
        offset,
        range_Ψ,
        range_Ω11,
        range_Ω12,
        range_Ω21,
        range_block,
    )
end

"""
    build_selfenergy_aux_layout(blocks)

Build per-block local auxiliary-state layouts and cumulative ranges in a future flattened
heterogeneous vector. Returns `(layouts, total_size)`.
"""
function build_selfenergy_aux_layout(blocks::AbstractVector{SelfEnergyBlock})
    layouts = SelfEnergyAuxBlockLayout[]
    next_offset = 1

    for block in blocks
        layout = SelfEnergyAuxBlockLayout(block, next_offset)
        push!(layouts, layout)
        next_offset += layout.size_block
    end

    total_size = next_offset - 1
    return layouts, total_size
end
