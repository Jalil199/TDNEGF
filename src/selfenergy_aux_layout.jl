"""
    SelfEnergyAuxBlockLayout

Linearization metadata for one block's `Ψ` sector inside the flattened auxiliary
state vector (excluding the leading global `ρ_ab` segment).

For block `i`, `Ψ_anλ` has shape `(Ns, Nc_i, N_λ_i)` and contributes
`size_Ψ = Ns * Nc_i * N_λ_i` contiguous entries.
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
    SelfEnergyAuxPairLayout

Linearization metadata for the ordered block pair `(i, j)` Ω sectors:
- `Ω11`: `(Nc_i, N_λ1_i, Nc_j, N_λ1_j)`
- `Ω12`: `(Nc_i, N_λ1_i, Nc_j, N_λ2_j)`
- `Ω21`: `(Nc_i, N_λ2_i, Nc_j, N_λ1_j)`

This mirrors the legacy monolithic ordering, but with per-block heterogeneous
sizes.
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

"""
    build_selfenergy_aux_layout(blocks)

Build the block-based auxiliary linearization (without `ρ_ab`):
1. all block `Ψ` segments in block order,
2. then all ordered-pair Ω segments in `(i, j)` nested-loop order.

`total_size` is the number of auxiliary entries after `ρ_ab` in the flattened
state vector used by `pointer_blocks`.
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
    SelfEnergyAuxLayout

Global layout object for the experimental block-based path.

It stores per-block and per-pair ranges in the auxiliary tail of the flattened
state. The full ODE state is `[vec(ρ_ab); aux_tail]`, so these ranges are
shifted by `length(vec(ρ_ab))` when creating runtime views.
"""
Base.@kwdef struct SelfEnergyAuxLayout
    block_layouts::Vector{SelfEnergyAuxBlockLayout}
    pair_layouts::Matrix{SelfEnergyAuxPairLayout}
    total_size::Int
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

The returned views follow the same logical ordering as the legacy pointer
layout, but with heterogeneous block sizes.
"""
@inline function pointer_blocks(
    vec::Vector{ComplexF64},
    dims_ρ_ab::NTuple{2,Int},
    layout::SelfEnergyAuxLayout,
)
    size_ρ_ab = prod(dims_ρ_ab)
    required_size = size_ρ_ab + layout.total_size
    length(vec) < required_size && throw(ArgumentError("vector length $(length(vec)) is smaller than required size $(required_size)"))

    # Leading segment is always the global density matrix.
    ρ_ab = reshape(view(vec, 1:size_ρ_ab), dims_ρ_ab)

    nblocks = length(layout.block_layouts)
    block_views = Vector{SelfEnergyAuxBlockPointers}(undef, nblocks)

    base = size_ρ_ab
    max_end = base
    for i in 1:nblocks
        bl = layout.block_layouts[i]
        # Layout ranges are defined in the auxiliary tail; shift by `base` to
        # map them onto full-state indices [vec(ρ_ab); aux_tail].
        range_Ψ = (first(bl.range_Ψ) + base):(last(bl.range_Ψ) + base)
        Ψ_anλ = reshape(view(vec, range_Ψ), (bl.Ns, bl.Nc, bl.N_λ))
        block_views[i] = SelfEnergyAuxBlockPointers(name = bl.name, Ψ_anλ = Ψ_anλ, range_Ψ = range_Ψ)
        max_end = max(max_end, last(range_Ψ))
    end

    pair_views = Matrix{SelfEnergyAuxPairPointers}(undef, nblocks, nblocks)
    for i in 1:nblocks
        for j in 1:nblocks
            pl = layout.pair_layouts[i, j]
            # Ordered pair (i,j) uses three Ω sectors with heterogeneous shapes.
            range_Ω11 = (first(pl.range_Ω11) + base):(last(pl.range_Ω11) + base)
            range_Ω12 = (first(pl.range_Ω12) + base):(last(pl.range_Ω12) + base)
            range_Ω21 = (first(pl.range_Ω21) + base):(last(pl.range_Ω21) + base)

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
            max_end = max(max_end, last(range_Ω21))
        end
    end

    # Keep a global consistency guard so layout construction remains reusable and
    # future extensions cannot silently drift from the flattened storage contract.
    max_end == required_size || throw(ArgumentError("layout sizes are inconsistent with flattened vector size"))

    return HeterogeneousAuxPointers(ρ_ab = ρ_ab, blocks = block_views, Ω_pairs = pair_views)
end
