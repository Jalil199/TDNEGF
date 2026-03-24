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

"""
Views for one heterogeneous auxiliary block inside a flattened vector.
"""
Base.@kwdef struct SelfEnergyAuxBlockPointers{
    TΨ<:AbstractArray{ComplexF64,3},
    TΩ11<:AbstractArray{ComplexF64,4},
    TΩ12<:AbstractArray{ComplexF64,4},
    TΩ21<:AbstractArray{ComplexF64,4},
}
    name::Symbol
    Ψ_anλ::TΨ
    Ω11::TΩ11
    Ω12::TΩ12
    Ω21::TΩ21
    range_Ψ::UnitRange{Int}
    range_Ω11::UnitRange{Int}
    range_Ω12::UnitRange{Int}
    range_Ω21::UnitRange{Int}
end

"""
Pointer-like views for the future heterogeneous auxiliary-state layout.
"""
Base.@kwdef struct HeterogeneousAuxPointers{Tρ<:AbstractArray{ComplexF64,2}, TB<:AbstractVector}
    ρ_ab::Tρ
    blocks::TB
end

"""
    pointer_blocks(vec, dims_ρ_ab, layouts)

Return zero-copy views into a flattened vector with:
- one global `ρ_ab` view
- one view bundle per heterogeneous auxiliary block (`Ψ`, `Ω11`, `Ω12`, `Ω21`)
"""
@inline function pointer_blocks(
    vec::Vector{ComplexF64},
    dims_ρ_ab::NTuple{2,Int},
    layouts::AbstractVector{SelfEnergyAuxBlockLayout},
)
    size_ρ_ab = prod(dims_ρ_ab)
    range_ρ_ab = 1:size_ρ_ab
    ρ_ab = reshape(view(vec, range_ρ_ab), dims_ρ_ab)

    aux_offset = size_ρ_ab
    block_views = Vector{SelfEnergyAuxBlockPointers}(undef, length(layouts))
    for (i, layout) in enumerate(layouts)
        shift = aux_offset + layout.offset - 1
        range_Ψ = (first(layout.range_Ψ) + shift):(last(layout.range_Ψ) + shift)
        range_Ω11 = (first(layout.range_Ω11) + shift):(last(layout.range_Ω11) + shift)
        range_Ω12 = (first(layout.range_Ω12) + shift):(last(layout.range_Ω12) + shift)
        range_Ω21 = (first(layout.range_Ω21) + shift):(last(layout.range_Ω21) + shift)

        Ψ_anλ = reshape(view(vec, range_Ψ), (layout.Ns, layout.Nc, layout.N_λ))
        Ω11 = reshape(view(vec, range_Ω11), (layout.Nc, layout.N_λ1, layout.Nc, layout.N_λ1))
        Ω12 = reshape(view(vec, range_Ω12), (layout.Nc, layout.N_λ1, layout.Nc, layout.N_λ2))
        Ω21 = reshape(view(vec, range_Ω21), (layout.Nc, layout.N_λ2, layout.Nc, layout.N_λ1))

        block_views[i] = SelfEnergyAuxBlockPointers(
            name = layout.name,
            Ψ_anλ = Ψ_anλ,
            Ω11 = Ω11,
            Ω12 = Ω12,
            Ω21 = Ω21,
            range_Ψ = range_Ψ,
            range_Ω11 = range_Ω11,
            range_Ω12 = range_Ω12,
            range_Ω21 = range_Ω21,
        )
    end

    return HeterogeneousAuxPointers(ρ_ab = ρ_ab, blocks = block_views)
end
