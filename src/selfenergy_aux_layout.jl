"""
Lightweight local layout for one heterogeneous self-energy auxiliary block.
Stores only Ψ for the source block.
"""
Base.@kwdef struct SelfEnergyAuxBlockLayout
    name::Symbol
    Ns::Int
    Nc::Int
    N_λ1::Int
    N_λ2::Int
    N_λ::Int
    size_Ψ::Int
    offset::Int
    range_Ψ::UnitRange{Int}
end

@inline function SelfEnergyAuxBlockLayout(block::SelfEnergyBlock, offset::Int)
    Ns = size(block.ξ_an, 1)
    Nc = block.Nc
    N_λ1 = block.N_λ1
    N_λ2 = block.N_λ2
    N_λ = block.N_λ

    size_Ψ = Ns * Nc * N_λ
    range_Ψ = offset:(offset + size_Ψ - 1)

    return SelfEnergyAuxBlockLayout(
        ;
        name = block.name,
        Ns,
        Nc,
        N_λ1,
        N_λ2,
        N_λ,
        size_Ψ,
        offset,
        range_Ψ,
    )
end

"""
Layout for Ω sectors that couple one source block `i` to one target block `j`.
"""
Base.@kwdef struct SelfEnergyAuxPairLayout
    source::Int
    target::Int
    source_name::Symbol
    target_name::Symbol
    Nc_i::Int
    N_λ1_i::Int
    N_λ2_i::Int
    Nc_j::Int
    N_λ1_j::Int
    N_λ2_j::Int
    size_Ω11::Int
    size_Ω12::Int
    size_Ω21::Int
    size_pair::Int
    offset::Int
    range_Ω11::UnitRange{Int}
    range_Ω12::UnitRange{Int}
    range_Ω21::UnitRange{Int}
    range_pair::UnitRange{Int}
end

@inline function SelfEnergyAuxPairLayout(
    source::Int,
    target::Int,
    source_block::SelfEnergyBlock,
    target_block::SelfEnergyBlock,
    offset::Int,
)
    Nc_i = source_block.Nc
    N_λ1_i = source_block.N_λ1
    N_λ2_i = source_block.N_λ2

    Nc_j = target_block.Nc
    N_λ1_j = target_block.N_λ1
    N_λ2_j = target_block.N_λ2

    size_Ω11 = Nc_i * N_λ1_i * Nc_j * N_λ1_j
    size_Ω12 = Nc_i * N_λ1_i * Nc_j * N_λ2_j
    size_Ω21 = Nc_i * N_λ2_i * Nc_j * N_λ1_j
    size_pair = size_Ω11 + size_Ω12 + size_Ω21

    range_Ω11 = offset:(offset + size_Ω11 - 1)
    range_Ω12 = (last(range_Ω11) + 1):(last(range_Ω11) + size_Ω12)
    range_Ω21 = (last(range_Ω12) + 1):(last(range_Ω12) + size_Ω21)
    range_pair = offset:(offset + size_pair - 1)

    return SelfEnergyAuxPairLayout(
        ;
        source,
        target,
        source_name = source_block.name,
        target_name = target_block.name,
        Nc_i,
        N_λ1_i,
        N_λ2_i,
        Nc_j,
        N_λ1_j,
        N_λ2_j,
        size_Ω11,
        size_Ω12,
        size_Ω21,
        size_pair,
        offset,
        range_Ω11,
        range_Ω12,
        range_Ω21,
        range_pair,
    )
end

Base.@kwdef struct SelfEnergyAuxLayout
    block_layouts::Vector{SelfEnergyAuxBlockLayout}
    pair_layouts::Matrix{SelfEnergyAuxPairLayout}
    total_size::Int
end

"""
    build_selfenergy_aux_layout(blocks)

Build per-block Ψ layouts and pairwise Ω layouts. Returns a `SelfEnergyAuxLayout`.
"""
function build_selfenergy_aux_layout(blocks::AbstractVector{SelfEnergyBlock})
    nblocks = length(blocks)
    block_layouts = Vector{SelfEnergyAuxBlockLayout}(undef, nblocks)

    next_offset = 1
    for i in 1:nblocks
        layout = SelfEnergyAuxBlockLayout(blocks[i], next_offset)
        block_layouts[i] = layout
        next_offset += layout.size_Ψ
    end

    pair_layouts = Matrix{SelfEnergyAuxPairLayout}(undef, nblocks, nblocks)
    for i in 1:nblocks
        for j in 1:nblocks
            layout = SelfEnergyAuxPairLayout(i, j, blocks[i], blocks[j], next_offset)
            pair_layouts[i, j] = layout
            next_offset += layout.size_pair
        end
    end

    total_size = next_offset - 1
    return SelfEnergyAuxLayout(block_layouts = block_layouts, pair_layouts = pair_layouts, total_size = total_size)
end

"""
Views for one heterogeneous auxiliary Ψ block.
"""
Base.@kwdef struct SelfEnergyAuxBlockPointers{TΨ<:AbstractArray{ComplexF64,3}}
    name::Symbol
    Ψ_anλ::TΨ
    range_Ψ::UnitRange{Int}
end

"""
Views for one heterogeneous pair Ω sector.
"""
Base.@kwdef struct SelfEnergyAuxPairPointers{
    TΩ11<:AbstractArray{ComplexF64,4},
    TΩ12<:AbstractArray{ComplexF64,4},
    TΩ21<:AbstractArray{ComplexF64,4},
}
    source::Int
    target::Int
    source_name::Symbol
    target_name::Symbol
    Ω11::TΩ11
    Ω12::TΩ12
    Ω21::TΩ21
    range_Ω11::UnitRange{Int}
    range_Ω12::UnitRange{Int}
    range_Ω21::UnitRange{Int}
end

"""
Pointer-like views for the heterogeneous auxiliary-state layout.
"""
Base.@kwdef struct HeterogeneousAuxPointers{
    Tρ<:AbstractArray{ComplexF64,2},
    TB<:AbstractVector,
    TP<:AbstractMatrix,
}
    ρ_ab::Tρ
    blocks::TB
    Ω_pairs::TP
end

"""
    pointer_blocks(vec, dims_ρ_ab, layout)

Return zero-copy views into a flattened vector with:
- one global `ρ_ab` view,
- one view per block for `Ψ`,
- one view per ordered block pair `(i,j)` for `Ω11`, `Ω12`, `Ω21`.
"""
@inline function pointer_blocks(
    vec::Vector{ComplexF64},
    dims_ρ_ab::NTuple{2,Int},
    layout::SelfEnergyAuxLayout,
)
    size_ρ_ab = prod(dims_ρ_ab)
    required_size = size_ρ_ab + layout.total_size
    length(vec) < required_size && throw(ArgumentError("vector length $(length(vec)) is smaller than required size $(required_size)"))

    ρ_ab = reshape(view(vec, 1:size_ρ_ab), dims_ρ_ab)

    nblocks = length(layout.block_layouts)
    block_views = Vector{SelfEnergyAuxBlockPointers}(undef, nblocks)

    next_idx = size_ρ_ab + 1
    for i in 1:nblocks
        bl = layout.block_layouts[i]
        range_Ψ = next_idx:(next_idx + bl.size_Ψ - 1)
        next_idx = last(range_Ψ) + 1

        Ψ_anλ = reshape(view(vec, range_Ψ), (bl.Ns, bl.Nc, bl.N_λ))
        block_views[i] = SelfEnergyAuxBlockPointers(name = bl.name, Ψ_anλ = Ψ_anλ, range_Ψ = range_Ψ)
    end

    pair_views = Matrix{SelfEnergyAuxPairPointers}(undef, nblocks, nblocks)
    for i in 1:nblocks
        for j in 1:nblocks
            pl = layout.pair_layouts[i, j]

            range_Ω11 = next_idx:(next_idx + pl.size_Ω11 - 1)
            range_Ω12 = (last(range_Ω11) + 1):(last(range_Ω11) + pl.size_Ω12)
            range_Ω21 = (last(range_Ω12) + 1):(last(range_Ω12) + pl.size_Ω21)
            next_idx = last(range_Ω21) + 1

            Ω11 = reshape(view(vec, range_Ω11), (pl.Nc_i, pl.N_λ1_i, pl.Nc_j, pl.N_λ1_j))
            Ω12 = reshape(view(vec, range_Ω12), (pl.Nc_i, pl.N_λ1_i, pl.Nc_j, pl.N_λ2_j))
            Ω21 = reshape(view(vec, range_Ω21), (pl.Nc_i, pl.N_λ2_i, pl.Nc_j, pl.N_λ1_j))

            pair_views[i, j] = SelfEnergyAuxPairPointers(
                source = pl.source,
                target = pl.target,
                source_name = pl.source_name,
                target_name = pl.target_name,
                Ω11 = Ω11,
                Ω12 = Ω12,
                Ω21 = Ω21,
                range_Ω11 = range_Ω11,
                range_Ω12 = range_Ω12,
                range_Ω21 = range_Ω21,
            )
        end
    end

    next_idx - 1 == required_size || throw(ArgumentError("layout sizes are inconsistent with flattened vector size"))

    return HeterogeneousAuxPointers(ρ_ab = ρ_ab, blocks = block_views, Ω_pairs = pair_views)
end
